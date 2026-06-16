"""Proxy metrics for local solver experiments.

These metrics are not the external benchmark score. They provide a stable local
signal for comparing solver variants before spending leaderboard runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoresearch_agent.solver_dev.parser import CandidateTable
from autoresearch_agent.solver_dev.validator import ValidationResult

PROXY_SCORE_WEIGHT = 0.5
PROXY_EXPECTED_FAILED_WEIGHT = 100.0
PROXY_EXPECTED_SUCCESS_CREDIT = 500.0
PROXY_UNASSIGNED_TASK_WEIGHT = 10_000.0
PROXY_INVALID_PENALTY = 1_000_000.0
PROXY_TIMEOUT_PENALTY = 1_000_000.0
PROXY_MISSING_PAIR_PENALTY = 100_000.0
PROXY_DISPATCHED_PAIR_PENALTY = 8.0
PROXY_DUPLICATE_ASSIGNMENT_PENALTY = 0.0
PROXY_BUNDLED_COVERED_TASK_PENALTY = -12.0
PROXY_LOW_WILLINGNESS_TARGET_SUCCESS_RATIO = 0.88
PROXY_LOW_WILLINGNESS_SHORTFALL_PENALTY = 250.0


@dataclass(frozen=True)
class SolutionMetrics:
    """Comparable local metrics for one solver output."""

    is_valid: bool
    assignment_count: int
    assigned_task_count: int
    unassigned_task_count: int
    task_coverage_ratio: float
    used_courier_count: int
    courier_usage_ratio: float
    dispatched_pair_count: int
    missing_candidate_pair_count: int
    single_task_assignment_count: int
    bundled_task_assignment_count: int
    duplicate_dispatch_assignment_count: int
    total_score: float
    average_score_per_assignment: float
    average_score_per_task: float
    total_willingness: float
    average_willingness: float
    expected_successful_task_count: float
    expected_failed_task_count: float
    expected_success_ratio: float
    score_per_expected_successful_task: float
    duplicate_dispatched_task_count: int
    bundled_covered_task_count: int
    risk_adjusted_proxy_score: float
    proxy_score: float


def calculate_solution_metrics(
    validation: ValidationResult,
    candidate_table: CandidateTable,
    *,
    case_id: str = "",
    timed_out: bool = False,
) -> SolutionMetrics:
    """Calculate local proxy metrics from a validated solver output."""
    assigned_tasks: set[str] = set()
    used_couriers: set[str] = set()
    bundled_covered_tasks: set[str] = set()
    task_success_probabilities: dict[str, list[float]] = {}
    dispatched_pair_count = 0
    missing_candidate_pair_count = 0
    single_task_assignment_count = 0
    bundled_task_assignment_count = 0
    duplicate_dispatch_assignment_count = 0
    total_score = 0.0
    total_willingness = 0.0

    for assignment in validation.assignments:
        assigned_tasks.update(assignment.task_id_list)
        used_couriers.update(assignment.courier_ids)

        if len(assignment.task_id_list) == 1:
            single_task_assignment_count += 1
        else:
            bundled_task_assignment_count += 1
            bundled_covered_tasks.update(assignment.task_id_list)

        if len(assignment.courier_ids) > 1:
            duplicate_dispatch_assignment_count += 1

        for courier_id in assignment.courier_ids:
            dispatched_pair_count += 1
            candidate = candidate_table.candidate_map.get(
                (assignment.task_id_list_str, courier_id)
            )
            if candidate is None:
                missing_candidate_pair_count += 1
                continue
            total_score += candidate.total_score
            total_willingness += candidate.willingness
            willingness = _clamp_probability(candidate.willingness)
            for task_id in assignment.task_id_list:
                task_success_probabilities.setdefault(task_id, []).append(willingness)

    assignment_count = len(validation.assignments)
    assigned_task_count = len(assigned_tasks)
    task_count = len(candidate_table.task_ids)
    courier_count = len(candidate_table.courier_ids)
    unassigned_task_count = max(task_count - assigned_task_count, 0)
    expected_successful_task_count = sum(
        _combined_success_probability(probabilities)
        for probabilities in task_success_probabilities.values()
    )
    expected_failed_task_count = max(task_count - expected_successful_task_count, 0.0)
    expected_success_ratio = (
        expected_successful_task_count / task_count if task_count else 0.0
    )
    score_per_expected_successful_task = (
        total_score / expected_successful_task_count
        if expected_successful_task_count
        else 0.0
    )
    duplicate_dispatched_task_count = sum(
        1
        for probabilities in task_success_probabilities.values()
        if len(probabilities) > 1
    )

    average_score_per_assignment = (
        total_score / assignment_count if assignment_count else 0.0
    )
    average_score_per_task = (
        total_score / assigned_task_count if assigned_task_count else 0.0
    )
    average_willingness = (
        total_willingness / dispatched_pair_count if dispatched_pair_count else 0.0
    )

    proxy_score = _calculate_proxy_score(
        is_valid=validation.is_valid,
        timed_out=timed_out,
        unassigned_task_count=unassigned_task_count,
        total_score=total_score,
        expected_successful_task_count=expected_successful_task_count,
        expected_failed_task_count=expected_failed_task_count,
        missing_candidate_pair_count=missing_candidate_pair_count,
        dispatched_pair_count=dispatched_pair_count,
        duplicate_dispatch_assignment_count=duplicate_dispatch_assignment_count,
        bundled_covered_task_count=len(bundled_covered_tasks),
        case_id=case_id,
    )
    return SolutionMetrics(
        is_valid=validation.is_valid,
        assignment_count=assignment_count,
        assigned_task_count=assigned_task_count,
        unassigned_task_count=unassigned_task_count,
        task_coverage_ratio=assigned_task_count / task_count if task_count else 0.0,
        used_courier_count=len(used_couriers),
        courier_usage_ratio=len(used_couriers) / courier_count
        if courier_count
        else 0.0,
        dispatched_pair_count=dispatched_pair_count,
        missing_candidate_pair_count=missing_candidate_pair_count,
        single_task_assignment_count=single_task_assignment_count,
        bundled_task_assignment_count=bundled_task_assignment_count,
        duplicate_dispatch_assignment_count=duplicate_dispatch_assignment_count,
        total_score=total_score,
        average_score_per_assignment=average_score_per_assignment,
        average_score_per_task=average_score_per_task,
        total_willingness=total_willingness,
        average_willingness=average_willingness,
        expected_successful_task_count=expected_successful_task_count,
        expected_failed_task_count=expected_failed_task_count,
        expected_success_ratio=expected_success_ratio,
        score_per_expected_successful_task=score_per_expected_successful_task,
        duplicate_dispatched_task_count=duplicate_dispatched_task_count,
        bundled_covered_task_count=len(bundled_covered_tasks),
        risk_adjusted_proxy_score=proxy_score,
        proxy_score=proxy_score,
    )


def metrics_to_serializable(metrics: SolutionMetrics) -> dict[str, Any]:
    """Convert metrics to JSON-serializable primitives."""
    return {
        "is_valid": metrics.is_valid,
        "assignment_count": metrics.assignment_count,
        "assigned_task_count": metrics.assigned_task_count,
        "unassigned_task_count": metrics.unassigned_task_count,
        "task_coverage_ratio": metrics.task_coverage_ratio,
        "used_courier_count": metrics.used_courier_count,
        "courier_usage_ratio": metrics.courier_usage_ratio,
        "dispatched_pair_count": metrics.dispatched_pair_count,
        "missing_candidate_pair_count": metrics.missing_candidate_pair_count,
        "single_task_assignment_count": metrics.single_task_assignment_count,
        "bundled_task_assignment_count": metrics.bundled_task_assignment_count,
        "duplicate_dispatch_assignment_count": metrics.duplicate_dispatch_assignment_count,
        "total_score": metrics.total_score,
        "average_score_per_assignment": metrics.average_score_per_assignment,
        "average_score_per_task": metrics.average_score_per_task,
        "total_willingness": metrics.total_willingness,
        "average_willingness": metrics.average_willingness,
        "expected_successful_task_count": metrics.expected_successful_task_count,
        "expected_failed_task_count": metrics.expected_failed_task_count,
        "expected_success_ratio": metrics.expected_success_ratio,
        "score_per_expected_successful_task": metrics.score_per_expected_successful_task,
        "duplicate_dispatched_task_count": metrics.duplicate_dispatched_task_count,
        "bundled_covered_task_count": metrics.bundled_covered_task_count,
        "risk_adjusted_proxy_score": metrics.risk_adjusted_proxy_score,
        "proxy_score": metrics.proxy_score,
    }


def _calculate_proxy_score(
    *,
    is_valid: bool,
    timed_out: bool,
    unassigned_task_count: int,
    total_score: float,
    expected_successful_task_count: float,
    expected_failed_task_count: float,
    missing_candidate_pair_count: int,
    dispatched_pair_count: int,
    duplicate_dispatch_assignment_count: int,
    bundled_covered_task_count: int,
    case_id: str,
) -> float:
    """Return the lower-is-better local comparison score."""
    return _calculate_weighted_proxy_score(
        is_valid=is_valid,
        timed_out=timed_out,
        unassigned_task_count=unassigned_task_count,
        total_score=total_score,
        expected_successful_task_count=expected_successful_task_count,
        expected_failed_task_count=expected_failed_task_count,
        missing_candidate_pair_count=missing_candidate_pair_count,
        dispatched_pair_count=dispatched_pair_count,
        duplicate_dispatch_assignment_count=duplicate_dispatch_assignment_count,
        bundled_covered_task_count=bundled_covered_task_count,
        case_id=case_id,
        score_weight=PROXY_SCORE_WEIGHT,
        expected_failed_weight=PROXY_EXPECTED_FAILED_WEIGHT,
        expected_success_credit=PROXY_EXPECTED_SUCCESS_CREDIT,
        dispatched_pair_penalty=PROXY_DISPATCHED_PAIR_PENALTY,
        duplicate_assignment_penalty=PROXY_DUPLICATE_ASSIGNMENT_PENALTY,
        bundle_covered_task_penalty=PROXY_BUNDLED_COVERED_TASK_PENALTY,
        low_willingness_target_success_ratio=(
            PROXY_LOW_WILLINGNESS_TARGET_SUCCESS_RATIO
        ),
        low_willingness_shortfall_penalty=(
            PROXY_LOW_WILLINGNESS_SHORTFALL_PENALTY
        ),
    )


def _calculate_weighted_proxy_score(
    *,
    is_valid: bool,
    timed_out: bool,
    unassigned_task_count: int,
    total_score: float,
    expected_successful_task_count: float,
    expected_failed_task_count: float,
    missing_candidate_pair_count: int,
    dispatched_pair_count: int,
    duplicate_dispatch_assignment_count: int,
    bundled_covered_task_count: int,
    score_weight: float,
    expected_failed_weight: float,
    expected_success_credit: float,
    case_id: str = "",
    dispatched_pair_penalty: float = 0.0,
    duplicate_assignment_penalty: float = 0.0,
    bundle_covered_task_penalty: float = 0.0,
    low_willingness_target_success_ratio: float = 0.88,
    low_willingness_shortfall_penalty: float = 0.0,
) -> float:
    """Return a lower-is-better weighted local comparison score."""
    invalid_penalty = PROXY_INVALID_PENALTY if not is_valid else 0.0
    timeout_penalty = PROXY_TIMEOUT_PENALTY if timed_out else 0.0
    missing_pair_penalty = PROXY_MISSING_PAIR_PENALTY * missing_candidate_pair_count
    unassigned_penalty = PROXY_UNASSIGNED_TASK_WEIGHT * unassigned_task_count
    task_count = expected_successful_task_count + expected_failed_task_count
    low_willingness_shortfall = 0.0
    if _is_low_willingness_case(case_id):
        target_successful = low_willingness_target_success_ratio * task_count
        low_willingness_shortfall = max(
            target_successful - expected_successful_task_count,
            0.0,
        )
    return (
        invalid_penalty
        + timeout_penalty
        + missing_pair_penalty
        + unassigned_penalty
        + score_weight * total_score
        + expected_failed_weight * expected_failed_task_count
        - expected_success_credit * expected_successful_task_count
        + dispatched_pair_penalty * dispatched_pair_count
        + duplicate_assignment_penalty * duplicate_dispatch_assignment_count
        + bundle_covered_task_penalty * bundled_covered_task_count
        + low_willingness_shortfall_penalty * low_willingness_shortfall
    )


def _combined_success_probability(probabilities: list[float]) -> float:
    """Combine independent dispatch probabilities for one task."""
    failure_probability = 1.0
    for probability in probabilities:
        failure_probability *= 1.0 - probability
    return 1.0 - failure_probability


def _clamp_probability(value: float) -> float:
    """Clamp willingness into probability bounds for local risk metrics."""
    return min(max(value, 0.0), 1.0)


def _is_low_willingness_case(case_id: str) -> bool:
    """Return whether a case id represents low-willingness pressure."""
    return "low_willingness" in case_id or "lowwill" in case_id
