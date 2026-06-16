"""Fixed solver-variant discovery and batch evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from autoresearch_agent.solver_dev.case_suite import SolverCase
from autoresearch_agent.solver_dev.metrics import (
    SolutionMetrics,
    metrics_to_serializable,
)
from autoresearch_agent.solver_dev.runner import (
    SolverRunResult,
    SolverSuiteResult,
    run_solver_case,
    run_solver_suite,
    solver_run_to_serializable,
    solver_suite_to_serializable,
    suite_aggregate_to_serializable,
)


@dataclass(frozen=True)
class VariantRunResult:
    """One evaluated solver variant."""

    variant_path: str
    run_result: SolverRunResult
    rank: int = 0


@dataclass(frozen=True)
class VariantBatchResult:
    """Batch result for a set of solver variants."""

    case_path: str
    results: list[VariantRunResult]
    best_variant_path: str
    best_metrics: SolutionMetrics | None


@dataclass(frozen=True)
class VariantSuiteRunResult:
    """One evaluated solver variant across a case suite."""

    variant_path: str
    suite_result: SolverSuiteResult
    rank: int = 0


@dataclass(frozen=True)
class VariantSuiteBatchResult:
    """Batch result for solver variants across a case suite."""

    results: list[VariantSuiteRunResult]
    best_variant_path: str
    best_aggregate_metrics: dict[str, Any]


def discover_solver_variants(variants_dir: str | Path) -> list[Path]:
    """Discover fixed solver variant files in stable path order."""
    root = Path(variants_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.py") if not path.name.startswith("_"))


def run_solver_variant_batch(
    variant_paths: Iterable[str | Path],
    case_path: str | Path,
    timeout_seconds: float,
) -> VariantBatchResult:
    """Run all solver variants and return ranked results."""
    raw_results = [
        VariantRunResult(
            variant_path=str(Path(path)),
            run_result=run_solver_case(
                str(path),
                str(case_path),
                timeout_seconds=timeout_seconds,
            ),
        )
        for path in variant_paths
    ]
    ranked = rank_variant_results(raw_results)
    best = next(
        (result for result in ranked if result.run_result.metrics.is_valid),
        None,
    )
    return VariantBatchResult(
        case_path=str(case_path),
        results=ranked,
        best_variant_path=best.variant_path if best else "",
        best_metrics=best.run_result.metrics if best else None,
    )


def run_solver_variant_suite_batch(
    variant_paths: Iterable[str | Path],
    cases: list[SolverCase],
    timeout_seconds: float,
) -> VariantSuiteBatchResult:
    """Run all solver variants across a local case suite and rank them."""
    raw_results = [
        VariantSuiteRunResult(
            variant_path=str(Path(path)),
            suite_result=run_solver_suite(
                str(path),
                cases,
                timeout_seconds=timeout_seconds,
            ),
        )
        for path in variant_paths
    ]
    ranked = rank_variant_suite_results(raw_results)
    best = next(
        (result for result in ranked if result.suite_result.aggregate_metrics.is_valid),
        None,
    )
    return VariantSuiteBatchResult(
        results=ranked,
        best_variant_path=best.variant_path if best else "",
        best_aggregate_metrics=suite_aggregate_to_serializable(
            best.suite_result.aggregate_metrics
        )
        if best
        else {},
    )


def rank_variant_results(results: Iterable[VariantRunResult]) -> list[VariantRunResult]:
    """Rank valid variants by ascending proxy score, then invalid variants last."""
    sorted_results = sorted(
        results,
        key=lambda result: (
            not result.run_result.metrics.is_valid,
            result.run_result.metrics.proxy_score,
            result.variant_path,
        ),
    )
    return [
        VariantRunResult(
            variant_path=result.variant_path,
            run_result=result.run_result,
            rank=index,
        )
        for index, result in enumerate(sorted_results, start=1)
    ]


def rank_variant_suite_results(
    results: Iterable[VariantSuiteRunResult],
) -> list[VariantSuiteRunResult]:
    """Rank suite-valid variants by aggregate proxy score, invalid variants last."""
    sorted_results = sorted(
        results,
        key=lambda result: (
            not result.suite_result.aggregate_metrics.is_valid,
            result.suite_result.aggregate_metrics.timeout_count,
            result.suite_result.aggregate_metrics.invalid_case_count,
            result.suite_result.aggregate_metrics.mean_proxy_score,
            result.variant_path,
        ),
    )
    return [
        VariantSuiteRunResult(
            variant_path=result.variant_path,
            suite_result=result.suite_result,
            rank=index,
        )
        for index, result in enumerate(sorted_results, start=1)
    ]


def variant_run_to_serializable(result: VariantRunResult) -> dict[str, Any]:
    """Convert a variant run to JSON-serializable primitives."""
    payload = solver_run_to_serializable(result.run_result)
    payload["variant_path"] = result.variant_path
    payload["rank"] = result.rank
    return payload


def variant_suite_run_to_serializable(
    result: VariantSuiteRunResult,
) -> dict[str, Any]:
    """Convert a variant suite run to JSON-serializable primitives."""
    payload = solver_suite_to_serializable(result.suite_result)
    payload["variant_path"] = result.variant_path
    payload["rank"] = result.rank
    return payload


def variant_batch_to_serializable(batch: VariantBatchResult) -> dict[str, Any]:
    """Convert a variant batch result to JSON-serializable primitives."""
    return {
        "case_path": batch.case_path,
        "best_variant_path": batch.best_variant_path,
        "best_metrics": metrics_to_serializable(batch.best_metrics)
        if batch.best_metrics
        else {},
        "variant_results": [
            variant_run_to_serializable(result) for result in batch.results
        ],
    }


def variant_suite_batch_to_serializable(
    batch: VariantSuiteBatchResult,
) -> dict[str, Any]:
    """Convert a variant suite batch result to JSON-serializable primitives."""
    return {
        "best_variant_path": batch.best_variant_path,
        "best_aggregate_metrics": batch.best_aggregate_metrics,
        "variant_results": [
            variant_suite_run_to_serializable(result) for result in batch.results
        ],
    }
