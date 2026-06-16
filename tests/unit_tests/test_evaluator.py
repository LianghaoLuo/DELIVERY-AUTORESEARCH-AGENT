from pathlib import Path

import autoresearch_agent.research.evaluator as evaluator_module
from autoresearch_agent.research.evaluator import evaluate_strategy_space
from autoresearch_agent.research.strategy_space import build_broad_strategy_configs


def test_evaluator_runs_local_strategy_space_on_robust_suite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        evaluator_module,
        "build_strategy_configs",
        lambda name, **kwargs: build_broad_strategy_configs(alphas=[90.0])[:1],
    )

    result = evaluate_strategy_space(
        search_space="broad_strategy",
        data_file="data/large_seed301.txt",
        baseline_solver="solvers/prior_solver.py",
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "variants",
        timeout_seconds=10.0,
        candidate_limit=1,
        provenance={"origin": "agent_loop", "selected_by": "autopilot"},
        hypothesis="unit-test local-only sweep",
        evidence_profile={"failure_mode": "low_expected_success"},
    )

    assert result.latest_record["label"] == "broad-strategy-sweep"
    assert result.latest_record["provenance"]["origin"] == "agent_loop"
    assert result.latest_record["evidence_profile"]["failure_mode"] == (
        "low_expected_success"
    )
    assert result.latest_record["baseline_suite"]["aggregate_metrics"]["case_count"] == 10
    assert result.baseline_suite["aggregate_metrics"]["case_count"] == 10
    assert result.batch_payload["best_variant_path"]
    assert result.best_variant_metrics["mean_proxy_score"]
    assert len(result.leaderboard_rows) == 1
    assert len(result.solver_candidates) == 1
    assert (tmp_path / "experiments" / "runs.jsonl").exists()
