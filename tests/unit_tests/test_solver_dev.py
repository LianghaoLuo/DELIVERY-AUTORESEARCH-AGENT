from pathlib import Path

import pytest

from autoresearch_agent.research.experiment_store import ExperimentStore
from autoresearch_agent.solver_dev import metrics as metrics_module
from autoresearch_agent.solver_dev.case_suite import (
    build_robust_case_suite,
)
from autoresearch_agent.solver_dev.metrics import calculate_solution_metrics
from autoresearch_agent.solver_dev.packager import package_solver
from autoresearch_agent.solver_dev.parser import (
    parse_candidate_table,
    summarize_candidate_table,
)
from autoresearch_agent.solver_dev.runner import (
    run_solver_case,
    run_solver_suite,
    run_solver_text,
    solver_run_to_serializable,
)
from autoresearch_agent.solver_dev.validator import validate_solution
from autoresearch_agent.solver_dev.variants import (
    discover_solver_variants,
    rank_variant_results,
    run_solver_variant_batch,
    run_solver_variant_suite_batch,
)


def test_parse_candidate_table_extracts_case_statistics() -> None:
    input_text = Path("data/large_seed301.txt").read_text()

    table = parse_candidate_table(input_text)
    summary = summarize_candidate_table(table)

    assert summary["candidate_count"] > 0
    assert summary["task_count"] > 0
    assert summary["courier_count"] > 0
    assert summary["single_task_candidate_count"] > 0
    assert summary["bundled_task_candidate_count"] > 0
    assert 0.0 < summary["bundled_candidate_ratio"] < 1.0


def test_validate_solution_rejects_duplicate_courier() -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    table = parse_candidate_table(input_text)
    first = table.candidates[0]
    second = table.candidates[1]
    result = [
        (first.task_id_list_str, [first.courier_id]),
        (second.task_id_list_str, [first.courier_id]),
    ]

    validation = validate_solution(result, table)

    assert not validation.is_valid
    assert any("assigned twice" in error for error in validation.errors)


def test_calculate_solution_metrics_for_valid_baseline() -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    table = parse_candidate_table(input_text)
    result = [
        (table.candidates[0].task_id_list_str, [table.candidates[0].courier_id]),
        (table.candidates[1].task_id_list_str, [table.candidates[1].courier_id]),
    ]
    validation = validate_solution(result, table)

    metrics = calculate_solution_metrics(validation, table)

    assert metrics.is_valid
    assert metrics.assignment_count == 2
    assert metrics.assigned_task_count > 0
    assert metrics.unassigned_task_count < len(table.task_ids)
    assert metrics.total_score > 0.0
    assert metrics.average_willingness > 0.0
    assert metrics.proxy_score > 0.0
    assert metrics.expected_successful_task_count > 0.0
    assert metrics.expected_failed_task_count > 0.0
    assert metrics.risk_adjusted_proxy_score == metrics.proxy_score


def test_expected_success_combines_duplicate_dispatch_probabilities() -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    table = parse_candidate_table(input_text)
    single_by_task = {}
    for candidate in table.candidates:
        if len(candidate.task_id_list) == 1:
            single_by_task.setdefault(candidate.task_id_list_str, []).append(candidate)
    candidates_for_task = next(
        candidates
        for candidates in single_by_task.values()
        if len({candidate.courier_id for candidate in candidates}) >= 2
    )
    first, second = candidates_for_task[0], candidates_for_task[1]
    result = [(first.task_id_list_str, [first.courier_id, second.courier_id])]
    validation = validate_solution(result, table)

    metrics = calculate_solution_metrics(validation, table)
    expected = 1.0 - (1.0 - first.willingness) * (1.0 - second.willingness)

    assert validation.is_valid
    assert metrics.duplicate_dispatched_task_count == 1
    assert metrics.expected_successful_task_count == pytest.approx(expected)


def test_bundled_candidate_contributes_expected_success_to_each_task() -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    table = parse_candidate_table(input_text)
    bundled = next(
        candidate for candidate in table.candidates if len(candidate.task_id_list) > 1
    )
    result = [(bundled.task_id_list_str, [bundled.courier_id])]
    validation = validate_solution(result, table)

    metrics = calculate_solution_metrics(validation, table)

    assert validation.is_valid
    assert metrics.bundled_covered_task_count == len(bundled.task_id_list)
    assert metrics.expected_successful_task_count == pytest.approx(
        bundled.willingness * len(bundled.task_id_list)
    )


def test_invalid_solution_scores_worse_than_valid_solution() -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    table = parse_candidate_table(input_text)
    first = table.candidates[0]
    valid = calculate_solution_metrics(
        validate_solution([(first.task_id_list_str, [first.courier_id])], table),
        table,
    )
    invalid = calculate_solution_metrics(
        validate_solution([("UNKNOWN_TASK", [first.courier_id])], table),
        table,
    )

    assert not invalid.is_valid
    assert invalid.proxy_score > valid.proxy_score


def test_lowwill_case_id_uses_low_willingness_shortfall_penalty() -> None:
    base_kwargs = {
        "is_valid": True,
        "timed_out": False,
        "unassigned_task_count": 0,
        "total_score": 100.0,
        "expected_successful_task_count": 5.0,
        "expected_failed_task_count": 5.0,
        "missing_candidate_pair_count": 0,
        "dispatched_pair_count": 1,
        "duplicate_dispatch_assignment_count": 0,
        "bundled_covered_task_count": 0,
        "score_weight": 1.0,
        "expected_failed_weight": 0.0,
        "expected_success_credit": 0.0,
        "low_willingness_shortfall_penalty": 100.0,
    }

    normal = metrics_module._calculate_weighted_proxy_score(
        **base_kwargs,
        case_id="robust_full_anchor",
    )
    robust_lowwill = metrics_module._calculate_weighted_proxy_score(
        **base_kwargs,
        case_id="robust_lowwill_sparse_20x34",
    )
    legacy_low_willingness = metrics_module._calculate_weighted_proxy_score(
        **base_kwargs,
        case_id="low_willingness_20_tasks",
    )

    assert robust_lowwill > normal
    assert legacy_low_willingness == pytest.approx(robust_lowwill)


def test_build_robust_case_suite_varies_topology_deterministically() -> None:
    input_text = Path("data/large_seed301.txt").read_text()

    first_suite = build_robust_case_suite(input_text, source_name="large_seed301")
    second_suite = build_robust_case_suite(input_text, source_name="large_seed301")

    assert [case.case_id for case in first_suite] == [
        "robust_full_anchor",
        "robust_lowwill_20_score_sensitive",
        "robust_lowwill_full_soft",
        "robust_large_lowwill_tail_soft",
        "robust_medium_hash_25x50",
        "robust_medium_sparse_25x38",
        "robust_lowwill_sparse_20x34",
        "robust_scarce_soft_40x30",
        "robust_scarce_neutral_40x26",
        "robust_small_hash_10x22",
    ]
    assert [case.weight for case in first_suite] == [
        1.0,
        2.0,
        0.5,
        0.1,
        0.6,
        1.2,
        0.1,
        0.05,
        0.1,
        0.1,
    ]
    assert first_suite == second_suite

    tables = {
        case.case_id: parse_candidate_table(case.input_text) for case in first_suite
    }
    full = tables["robust_full_anchor"]
    sparse = tables["robust_medium_sparse_25x38"]
    scarce = tables["robust_scarce_neutral_40x26"]
    lowwill = tables["robust_lowwill_sparse_20x34"]

    assert len(first_suite) == 10
    assert len(sparse.candidates) < len(full.candidates)
    assert len(sparse.courier_ids) < len(full.courier_ids)
    assert len(scarce.courier_ids) == 26
    assert len(scarce.candidates) < len(full.candidates)
    assert _average_willingness(lowwill) < _average_willingness(full)
    for table in tables.values():
        assert table.candidates
        assert table.task_ids
        for task_id in table.task_ids:
            assert any(
                task_id in candidate.task_id_list for candidate in table.candidates
            )


def _average_willingness(table) -> float:
    """Return average candidate willingness for a parsed case table."""
    return sum(candidate.willingness for candidate in table.candidates) / len(
        table.candidates
    )


def test_run_solver_case_executes_and_validates_baseline() -> None:
    result = run_solver_case(
        "solvers/solver.py",
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )

    assert result.error == ""
    assert not result.timed_out
    assert result.process_exit_code == 0
    assert result.solver_sha256
    assert result.validation.is_valid, result.validation.errors
    assert result.validation.assigned_task_count > 0
    assert result.metrics.is_valid
    assert result.metrics.assigned_task_count == result.validation.assigned_task_count
    assert result.metrics.unassigned_task_count == 0
    assert result.metrics.task_coverage_ratio == 1.0
    assert result.metrics.total_score > 0.0
    assert isinstance(result.output, list)


def test_run_solver_text_captures_stdout_and_exception(tmp_path: Path) -> None:
    solver = tmp_path / "noisy_bad_solver.py"
    solver.write_text(
        "def solve(input_text):\n"
        "    print('hello from solver')\n"
        "    raise RuntimeError('boom')\n"
    )
    input_text = Path("data/large_seed301.txt").read_text()

    result = run_solver_text(
        str(solver),
        input_text,
        case_id="noisy",
        timeout_seconds=1.0,
    )

    assert not result.validation.is_valid
    assert "RuntimeError: boom" in result.error
    assert result.stdout == "hello from solver\n"


def test_run_solver_text_enforces_timeout(tmp_path: Path) -> None:
    solver = tmp_path / "slow_solver.py"
    solver.write_text(
        "import time\ndef solve(input_text):\n    time.sleep(5)\n    return []\n"
    )
    input_text = Path("data/large_seed301.txt").read_text()

    result = run_solver_text(
        str(solver),
        input_text,
        case_id="slow",
        timeout_seconds=0.05,
    )

    assert result.timed_out
    assert not result.validation.is_valid
    assert "TimeoutError" in result.error


def test_solver_run_to_serializable_contains_validation_and_metrics() -> None:
    result = run_solver_case(
        "solvers/solver.py",
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )

    serialized = solver_run_to_serializable(result)

    assert serialized["validation"]["is_valid"]
    assert serialized["metrics"]["assigned_task_count"] == 40
    assert "expected_success_ratio" in serialized["metrics"]
    assert "legacy_proxy_v2_score" not in serialized["metrics"]
    assert "solver_sha256" in serialized
    assert serialized["output_count"] == 40
    assert serialized["output_preview"]


def test_experiment_store_writes_jsonl_and_markdown(tmp_path: Path) -> None:
    result = run_solver_case(
        "solvers/solver.py",
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )
    store = ExperimentStore(root_dir=tmp_path)

    record = store.append_solver_run(
        result,
        label="baseline-test",
        notes="unit test record",
    )

    records = store.load_records()
    markdown = store.markdown_path.read_text()

    assert len(records) == 1
    assert records[0]["run_id"] == record["run_id"]
    assert record["label"] == "baseline-test"
    assert records[0]["run"]["metrics"]["assigned_task_count"] == 40
    assert "baseline-test" in markdown
    assert "Proxy score" in markdown


def test_package_solver_copies_candidate_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    destination = tmp_path / "solvers" / "agent_suggested_solver.py"
    source.write_text("def solve(input_text):\n    return []\n", encoding="utf-8")

    packaged = package_solver(str(source), str(destination))

    assert packaged == str(destination)
    assert destination.read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_run_solver_suite_aggregates_robust_cases() -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    cases = build_robust_case_suite(input_text, source_name="large_seed301")

    result = run_solver_suite(
        "solvers/solver.py",
        cases,
        timeout_seconds=10.0,
    )

    assert result.aggregate_metrics.case_count == 10
    assert result.aggregate_metrics.is_valid
    assert result.aggregate_metrics.timeout_count == 0
    assert result.aggregate_metrics.mean_expected_success_ratio > 0.0
    assert result.aggregate_metrics.worst_case_id


def test_discover_solver_variants_returns_stable_order(tmp_path: Path) -> None:
    variants_dir = tmp_path / "variants"
    variants_dir.mkdir()
    (variants_dir / "b.py").write_text("def solve(input_text):\n    return []\n")
    (variants_dir / "a.py").write_text("def solve(input_text):\n    return []\n")
    (variants_dir / "_skip.py").write_text("def solve(input_text):\n    return []\n")

    paths = discover_solver_variants(variants_dir)

    assert [path.name for path in paths] == [
        "a.py",
        "b.py",
    ]


def test_prior_solver_is_valid() -> None:
    paths = [Path("solvers/prior_solver.py")]

    for path in paths:
        result = run_solver_case(
            str(path),
            "data/large_seed301.txt",
            timeout_seconds=10.0,
        )
        assert result.validation.is_valid, (path, result.validation.errors)
        assert result.metrics.proxy_score


def test_run_solver_variant_batch_ranks_valid_variants() -> None:
    paths = _baseline_variant_paths()

    batch = run_solver_variant_batch(
        paths,
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )

    assert len(batch.results) == len(paths)
    assert [result.rank for result in batch.results] == [1, 2]
    assert batch.best_variant_path
    assert batch.best_metrics is not None
    assert batch.best_metrics.proxy_score


def test_run_solver_variant_suite_batch_ranks_valid_variants() -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    cases = build_robust_case_suite(input_text, source_name="large_seed301")
    paths = _baseline_variant_paths()

    batch = run_solver_variant_suite_batch(
        paths,
        cases,
        timeout_seconds=10.0,
    )

    assert len(batch.results) == len(paths)
    assert [result.rank for result in batch.results] == [1, 2]
    assert batch.best_variant_path
    assert batch.best_aggregate_metrics["mean_proxy_score"]


def test_rank_variant_results_puts_invalid_results_last(tmp_path: Path) -> None:
    invalid_solver = tmp_path / "invalid_solver.py"
    invalid_solver.write_text("def solve(input_text):\n    return 'bad'\n")
    paths = [
        invalid_solver,
        Path("solvers/solver.py"),
    ]

    batch = run_solver_variant_batch(
        paths,
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )
    ranked = rank_variant_results(batch.results)

    assert ranked[0].run_result.metrics.is_valid
    assert not ranked[-1].run_result.metrics.is_valid


def test_run_solver_variant_suite_batch_puts_invalid_results_last(
    tmp_path: Path,
) -> None:
    invalid_solver = tmp_path / "invalid_solver.py"
    invalid_solver.write_text("def solve(input_text):\n    return 'bad'\n")
    input_text = Path("data/large_seed301.txt").read_text()
    cases = build_robust_case_suite(input_text, source_name="large_seed301")
    paths = [
        invalid_solver,
        Path("solvers/solver.py"),
    ]

    batch = run_solver_variant_suite_batch(
        paths,
        cases,
        timeout_seconds=10.0,
    )

    assert batch.results[0].suite_result.aggregate_metrics.is_valid
    assert not batch.results[-1].suite_result.aggregate_metrics.is_valid


def test_experiment_store_writes_variant_batch(tmp_path: Path) -> None:
    baseline = run_solver_case(
        "solvers/solver.py",
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )
    batch = run_solver_variant_batch(
        _baseline_variant_paths(),
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )
    store = ExperimentStore(root_dir=tmp_path)

    record = store.append_variant_batch(
        batch,
        label="variant-batch-test",
        baseline_result=baseline,
        notes="unit test batch",
    )
    markdown = store.markdown_path.read_text()

    assert record["best_variant_path"] == batch.best_variant_path
    assert record["baseline_metrics"]["assigned_task_count"] == 40
    assert len(record["batch"]["variant_results"]) == 2
    assert "variant-batch-test" in markdown
    assert "Variant" in markdown


def test_experiment_store_writes_suite_and_variant_suite_batch(tmp_path: Path) -> None:
    input_text = Path("data/large_seed301.txt").read_text()
    cases = build_robust_case_suite(input_text, source_name="large_seed301")
    baseline = run_solver_suite(
        "solvers/solver.py",
        cases,
        timeout_seconds=10.0,
    )
    batch = run_solver_variant_suite_batch(
        _baseline_variant_paths(),
        cases,
        timeout_seconds=10.0,
    )
    store = ExperimentStore(root_dir=tmp_path)

    suite_record = store.append_solver_suite_run(
        baseline,
        label="suite-test",
        notes="unit test suite",
    )
    batch_record = store.append_variant_suite_batch(
        batch,
        label="variant-suite-test",
        baseline_result=baseline,
        notes="unit test variant suite",
    )
    markdown = store.markdown_path.read_text()

    assert suite_record["suite"]["aggregate_metrics"]["case_count"] == 10
    assert batch_record["best_variant_path"] == batch.best_variant_path
    assert len(batch_record["batch"]["variant_results"]) == 2
    assert "suite-test" in markdown
    assert "variant-suite-test" in markdown


def _baseline_variant_paths() -> list[Path]:
    """Return stable local solver files for variant-ranking tests."""
    return [
        Path("solvers/solver.py"),
        Path("solvers/prior_solver.py"),
    ]
