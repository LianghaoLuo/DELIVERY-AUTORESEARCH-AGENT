"""Unified local evaluation for AutoResearch strategy sweeps.

This module intentionally uses only the robust local case suite and proxy
metrics. External benchmark score records are outside the agent runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from autoresearch_agent.research.experiment_store import ExperimentStore
from autoresearch_agent.research.strategy_space import (
    build_strategy_configs,
    default_strategy_output_dir,
    materialize_strategy_configs,
    parse_strategy_metadata_from_variant_path,
    select_strategy_solver_candidates,
    strategy_search_leaderboard_rows,
    strategy_sweep_label,
    strategy_sweep_notes,
)
from autoresearch_agent.solver_dev.case_suite import (
    CaseSuiteName,
    SolverCase,
    build_case_suite,
)
from autoresearch_agent.solver_dev.runner import (
    SolverSuiteResult,
    run_solver_suite,
    solver_suite_to_serializable,
)
from autoresearch_agent.solver_dev.variants import (
    VariantSuiteBatchResult,
    run_solver_variant_suite_batch,
    variant_suite_batch_to_serializable,
)


@dataclass(frozen=True)
class LocalEvaluationResult:
    """Serializable payloads from one local AutoResearch evaluation."""

    latest_record: dict[str, Any]
    baseline_suite: dict[str, Any]
    baseline_full_case: dict[str, Any]
    batch_payload: dict[str, Any]
    leaderboard_rows: list[dict[str, Any]]
    solver_candidates: list[dict[str, Any]]
    best_variant_path: str
    best_variant_metrics: dict[str, Any]


def evaluate_strategy_space(
    *,
    search_space: str,
    data_file: str | Path,
    baseline_solver: str,
    experiments_dir: str | Path,
    output_dir: str | Path | None = None,
    timeout_seconds: float = 10.0,
    candidate_limit: int = 4,
    provenance: dict[str, Any] | None = None,
    hypothesis: str = "",
    alphas: Sequence[float] | None = None,
    selected_config_ids: Sequence[str] | None = None,
    inline_configs: Sequence[dict[str, Any]] | None = None,
    evidence_profile: dict[str, Any] | None = None,
    case_suite: CaseSuiteName = "robust",
) -> LocalEvaluationResult:
    """Generate, run, rank, and persist one registered strategy-space sweep."""
    cases = _build_cases(data_file, case_suite=case_suite)
    baseline_suite = run_solver_suite(
        baseline_solver,
        cases,
        timeout_seconds=timeout_seconds,
    )
    configs = build_strategy_configs(
        search_space,
        alphas=alphas,
        selected_config_ids=selected_config_ids,
        inline_configs=inline_configs,
    )
    variant_paths = materialize_strategy_configs(
        configs,
        output_dir or default_strategy_output_dir(search_space),
    )
    batch = run_solver_variant_suite_batch(
        variant_paths,
        cases,
        timeout_seconds=timeout_seconds,
    )
    store = ExperimentStore(root_dir=Path(experiments_dir))
    record = store.append_variant_suite_batch(
        batch,
        label=strategy_sweep_label(search_space),
        baseline_result=baseline_suite,
        notes=strategy_sweep_notes(search_space, hypothesis=hypothesis),
        provenance=provenance,
        baseline_suite=solver_suite_to_serializable(baseline_suite),
        evidence_profile=evidence_profile,
    )
    batch_payload = variant_suite_batch_to_serializable(batch)
    return _evaluation_result_from_batch(
        record=record,
        baseline_suite=baseline_suite,
        batch=batch,
        batch_payload=batch_payload,
        candidate_limit=candidate_limit,
    )


def evaluate_stop_search_candidate(
    *,
    solver_path: str,
    data_file: str | Path,
    baseline_solver: str,
    experiments_dir: str | Path,
    timeout_seconds: float = 10.0,
    provenance: dict[str, Any] | None = None,
    evidence_profile: dict[str, Any] | None = None,
    case_suite: CaseSuiteName = "robust",
) -> LocalEvaluationResult:
    """Re-run a recommended solver as stop-search evidence."""
    cases = _build_cases(data_file, case_suite=case_suite)
    baseline_suite = run_solver_suite(
        baseline_solver,
        cases,
        timeout_seconds=timeout_seconds,
    )
    stop_suite = run_solver_suite(
        solver_path,
        cases,
        timeout_seconds=timeout_seconds,
    )
    store = ExperimentStore(root_dir=Path(experiments_dir))
    record = store.append_solver_suite_run(
        stop_suite,
        label="graph-auto-stop-evidence",
        notes=(
            "Autopilot stop-search evidence run for the current best local "
            "strategy candidate."
        ),
        provenance=provenance,
        baseline_suite=solver_suite_to_serializable(baseline_suite),
        evidence_profile=evidence_profile,
    )
    stop_payload = solver_suite_to_serializable(stop_suite)
    metadata = parse_strategy_metadata_from_variant_path(solver_path)
    batch_payload = {
        "best_variant_path": solver_path,
        "best_aggregate_metrics": stop_payload["aggregate_metrics"],
        "variant_results": [
            {
                "rank": 1,
                "variant_path": solver_path,
                "output_signature": "",
                **stop_payload,
            }
        ],
    }
    candidates = [
        {
            "reason": "stop_search_recommended",
            "rank": 1,
            "variant_path": solver_path,
            "family": metadata["family"],
            "intent": metadata["intent"],
            "pipeline": metadata["pipeline"],
            "params": metadata["params"],
            "mean_proxy_score": stop_payload["aggregate_metrics"]["mean_proxy_score"],
            "output_signature": "",
        }
    ]
    best_metrics = dict(stop_payload["aggregate_metrics"])
    best_metrics["proxy_score"] = best_metrics.get("mean_proxy_score", 0.0)
    return LocalEvaluationResult(
        latest_record=record,
        baseline_suite=solver_suite_to_serializable(baseline_suite),
        baseline_full_case=_select_full_case_result(stop_payload["case_results"]),
        batch_payload=batch_payload,
        leaderboard_rows=[],
        solver_candidates=candidates,
        best_variant_path=solver_path,
        best_variant_metrics=best_metrics,
    )


def _build_cases(
    data_file: str | Path,
    *,
    case_suite: CaseSuiteName = "robust",
) -> list[SolverCase]:
    """Build the requested local case suite from one candidate-table file."""
    data_path = Path(data_file)
    input_text = data_path.read_text()
    return build_case_suite(
        input_text,
        source_name=str(data_path),
        suite_name=case_suite,
    )


def _evaluation_result_from_batch(
    *,
    record: dict[str, Any],
    baseline_suite: SolverSuiteResult,
    batch: VariantSuiteBatchResult,
    batch_payload: dict[str, Any],
    candidate_limit: int,
) -> LocalEvaluationResult:
    """Convert a variant-suite batch into the graph-facing result payload."""
    baseline_payload = solver_suite_to_serializable(baseline_suite)
    best_metrics = dict(batch_payload["best_aggregate_metrics"])
    if "mean_proxy_score" in best_metrics:
        best_metrics["proxy_score"] = best_metrics["mean_proxy_score"]
    return LocalEvaluationResult(
        latest_record=record,
        baseline_suite=baseline_payload,
        baseline_full_case=_select_full_case_result(baseline_payload["case_results"]),
        batch_payload=batch_payload,
        leaderboard_rows=strategy_search_leaderboard_rows(batch),
        solver_candidates=select_strategy_solver_candidates(
            batch,
            limit=candidate_limit,
        ),
        best_variant_path=batch_payload["best_variant_path"],
        best_variant_metrics=best_metrics,
    )


def _select_full_case_result(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the suite result for the full source case."""
    for result in case_results:
        if result.get("case_id") == "full":
            return result
    return case_results[0] if case_results else {}
