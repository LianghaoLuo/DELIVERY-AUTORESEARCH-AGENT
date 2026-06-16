"""Failure-mode evidence extraction for local AutoResearch decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FailureMode = Literal[
    "low_expected_success",
    "scarce_courier_pressure",
    "bundle_underuse",
    "over_duplicate_cost",
    "high_score_cost",
    "invalid_or_timeout",
    "stagnation",
    "bootstrap",
]


@dataclass(frozen=True)
class EvidenceProfile:
    """Structured local evidence used to choose the next research action."""

    failure_mode: FailureMode
    affected_cases: list[str]
    severity: float
    strategy_family: str
    param_region: dict[str, Any] = field(default_factory=dict)
    evidence_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def evidence_profile_to_dict(profile: EvidenceProfile) -> dict[str, Any]:
    """Convert an evidence profile to JSON-serializable primitives."""
    return {
        "failure_mode": profile.failure_mode,
        "affected_cases": list(profile.affected_cases),
        "severity": profile.severity,
        "strategy_family": profile.strategy_family,
        "param_region": dict(profile.param_region),
        "evidence_score": profile.evidence_score,
        "reasons": list(profile.reasons),
    }


def build_evidence_profile(
    records: list[dict[str, Any]],
    history_summary: dict[str, Any] | None = None,
) -> EvidenceProfile:
    """Infer the dominant local failure mode from append-only experiment records."""
    history_summary = history_summary or {}
    variant_records = [
        record
        for record in records
        if isinstance(record.get("batch"), dict)
        and isinstance(record.get("batch", {}).get("variant_results"), list)
    ]
    if not variant_records:
        return EvidenceProfile(
            failure_mode="bootstrap",
            affected_cases=[],
            severity=1.0,
            strategy_family="broad_strategy",
            param_region={"profile": "broad"},
            evidence_score=1.0,
            reasons=["No local variant-suite evidence is available yet."],
        )

    recent_records = variant_records[-3:]
    latest_record = variant_records[-1]
    latest_result = _best_valid_result(latest_record) or _first_result(latest_record)
    aggregate = _aggregate_metrics(latest_result)
    case_results = _case_results(latest_result)

    if _has_invalid_or_timeout(recent_records):
        return EvidenceProfile(
            failure_mode="invalid_or_timeout",
            affected_cases=_affected_invalid_cases(case_results),
            severity=1.0,
            strategy_family=_fallback_low_complexity_family(history_summary),
            param_region=_low_complexity_region(),
            evidence_score=1.0,
            reasons=[
                "Recent local evidence contains invalid output or timeout risk.",
                "Next sweep should reduce search complexity and duplicate intensity.",
            ],
        )

    scarce_cases = _scarce_pressure_cases(aggregate, case_results)
    if scarce_cases:
        severity = _max_case_value(case_results, "courier_usage_ratio", default=0.95)
        return EvidenceProfile(
            failure_mode="scarce_courier_pressure",
            affected_cases=scarce_cases,
            severity=severity,
            strategy_family="bundle_merge_duplicate",
            param_region=_bundle_merge_region(),
            evidence_score=severity,
            reasons=[
                "Worst-case evidence points to scarce courier capacity.",
                "Bundle-aware primary assignment can free couriers before duplicate dispatch.",
            ],
        )

    low_success_cases = _low_success_cases(aggregate, case_results)
    if low_success_cases:
        if int(history_summary.get("risk_tier_duplicate_sweep_count", 0)) > 0:
            strategy_family = "low_willingness_deep_duplicate"
        else:
            strategy_family = "risk_tier_duplicate"
        severity = 1.0 - _min_expected_success(aggregate, case_results)
        return EvidenceProfile(
            failure_mode="low_expected_success",
            affected_cases=low_success_cases,
            severity=severity,
            strategy_family=strategy_family,
            param_region=_risk_duplicate_region(strategy_family),
            evidence_score=severity,
            reasons=[
                "Local cases still show low expected task success.",
                "Risk-aware duplicate dispatch directly targets low-probability tasks.",
            ],
        )

    bundle_cases = _bundle_underuse_cases(case_results)
    if bundle_cases:
        severity = 0.7
        return EvidenceProfile(
            failure_mode="bundle_underuse",
            affected_cases=bundle_cases,
            severity=severity,
            strategy_family="bundle_merge_duplicate",
            param_region=_bundle_merge_region(),
            evidence_score=severity,
            reasons=[
                "Bundle-dense evidence shows little or no bundle usage.",
                "Bundle-merge search should test whether pairing singles improves courier usage.",
            ],
        )

    duplicate_cases = _over_duplicate_cases(case_results)
    if duplicate_cases:
        severity = _max_duplicate_ratio(case_results)
        return EvidenceProfile(
            failure_mode="over_duplicate_cost",
            affected_cases=duplicate_cases,
            severity=severity,
            strategy_family="risk_tier_duplicate",
            param_region=_conservative_duplicate_region(),
            evidence_score=severity,
            reasons=[
                "Duplicate dispatch appears expensive relative to the success gain.",
                "A conservative duplicate region limits candidate risk.",
            ],
        )

    high_score_cases = _high_score_cases(aggregate, case_results)
    if high_score_cases:
        severity = 0.55
        return EvidenceProfile(
            failure_mode="high_score_cost",
            affected_cases=high_score_cases,
            severity=severity,
            strategy_family="local_improve",
            param_region=_local_improve_region(),
            evidence_score=severity,
            reasons=[
                "Expected success is acceptable, but score cost remains high.",
                "Local replacement can seek lower-score assignments without broad exploration.",
            ],
        )

    if _has_stagnation(history_summary):
        recommended = _latest_best_path(history_summary)
        return EvidenceProfile(
            failure_mode="stagnation",
            affected_cases=_dominant_worst_cases(history_summary),
            severity=0.2,
            strategy_family="stop_search",
            param_region={"recommended_solver_path": recommended},
            evidence_score=0.2,
            reasons=[
                "Recent sweeps repeat the best output signature.",
                "No strong remaining failure mode is visible in local evidence.",
            ],
        )

    return EvidenceProfile(
        failure_mode="low_expected_success",
        affected_cases=[str(aggregate.get("worst_case_id", "full") or "full")],
        severity=0.35,
        strategy_family="risk_tier_duplicate",
        param_region=_risk_duplicate_region("risk_tier_duplicate"),
        evidence_score=0.35,
        reasons=[
            "Local evidence is valid but not yet conclusive.",
            "Risk-tier duplicate dispatch is the next targeted probe.",
        ],
    )


def _best_valid_result(record: dict[str, Any]) -> dict[str, Any]:
    for result in _record_results(record):
        aggregate = _aggregate_metrics(result)
        if bool(aggregate.get("is_valid", False)):
            return result
    return {}


def _first_result(record: dict[str, Any]) -> dict[str, Any]:
    results = _record_results(record)
    return results[0] if results else {}


def _record_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    batch = record.get("batch", {})
    results = batch.get("variant_results", []) if isinstance(batch, dict) else []
    return results if isinstance(results, list) else []


def _aggregate_metrics(result: dict[str, Any]) -> dict[str, Any]:
    aggregate = result.get("aggregate_metrics", {})
    return aggregate if isinstance(aggregate, dict) else {}


def _case_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    cases = result.get("case_results", [])
    return cases if isinstance(cases, list) else []


def _case_metrics(case_result: dict[str, Any]) -> dict[str, Any]:
    metrics = case_result.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _has_invalid_or_timeout(records: list[dict[str, Any]]) -> bool:
    for record in records:
        for result in _record_results(record):
            aggregate = _aggregate_metrics(result)
            if not bool(aggregate.get("is_valid", False)):
                return True
            if int(aggregate.get("timeout_count", 0)) > 0:
                return True
            if int(aggregate.get("invalid_case_count", 0)) > 0:
                return True
    return False


def _affected_invalid_cases(case_results: list[dict[str, Any]]) -> list[str]:
    cases = [
        str(case.get("case_id", ""))
        for case in case_results
        if bool(case.get("timed_out", False))
        or not bool(_case_metrics(case).get("is_valid", True))
    ]
    return [case for case in cases if case] or ["suite"]


def _scarce_pressure_cases(
    aggregate: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> list[str]:
    cases = []
    worst_case_id = str(aggregate.get("worst_case_id", ""))
    if "scarce" in worst_case_id:
        cases.append(worst_case_id)
    for case in case_results:
        case_id = str(case.get("case_id", ""))
        metrics = _case_metrics(case)
        if "scarce" in case_id:
            cases.append(case_id)
            continue
        if float(metrics.get("courier_usage_ratio", 0.0)) >= 0.95:
            cases.append(case_id or "unknown")
    return _unique(cases)


def _low_success_cases(
    aggregate: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> list[str]:
    cases = []
    mean_success = float(aggregate.get("mean_expected_success_ratio", 1.0))
    if mean_success < 0.89:
        cases.append(str(aggregate.get("worst_case_id", "suite") or "suite"))
    for case in case_results:
        case_id = str(case.get("case_id", ""))
        success = float(_case_metrics(case).get("expected_success_ratio", 1.0))
        if success < 0.84 or "low_willingness" in case_id:
            cases.append(case_id or "unknown")
    return _unique(cases)


def _bundle_underuse_cases(case_results: list[dict[str, Any]]) -> list[str]:
    cases = []
    for case in case_results:
        case_id = str(case.get("case_id", ""))
        metrics = _case_metrics(case)
        if "bundle" not in case_id:
            continue
        if int(metrics.get("bundled_task_assignment_count", 0)) == 0:
            cases.append(case_id)
    return _unique(cases)


def _over_duplicate_cases(case_results: list[dict[str, Any]]) -> list[str]:
    cases = []
    for case in case_results:
        metrics = _case_metrics(case)
        assignment_count = max(int(metrics.get("assignment_count", 0)), 1)
        duplicate_count = int(metrics.get("duplicate_dispatch_assignment_count", 0))
        total_score = float(metrics.get("total_score", 0.0))
        success = float(metrics.get("expected_success_ratio", 0.0))
        if duplicate_count / assignment_count >= 0.45 and success >= 0.9:
            cases.append(str(case.get("case_id", "unknown")))
        if duplicate_count > 0 and total_score / assignment_count >= 28.0:
            cases.append(str(case.get("case_id", "unknown")))
    return _unique(cases)


def _high_score_cases(
    aggregate: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> list[str]:
    cases: list[str] = []
    if float(aggregate.get("mean_expected_success_ratio", 0.0)) < 0.9:
        return cases
    for case in case_results:
        metrics = _case_metrics(case)
        score_per_success = float(
            metrics.get("score_per_expected_successful_task", 0.0)
        )
        if score_per_success >= 28.0:
            cases.append(str(case.get("case_id", "unknown")))
    return _unique(cases)


def _min_expected_success(
    aggregate: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> float:
    values = [float(aggregate.get("mean_expected_success_ratio", 1.0))]
    values.extend(
        float(_case_metrics(case).get("expected_success_ratio", 1.0))
        for case in case_results
    )
    return min(values) if values else 1.0


def _max_case_value(
    case_results: list[dict[str, Any]],
    key: str,
    *,
    default: float,
) -> float:
    values = [float(_case_metrics(case).get(key, 0.0)) for case in case_results]
    return max(values) if values else default


def _max_duplicate_ratio(case_results: list[dict[str, Any]]) -> float:
    ratios = []
    for case in case_results:
        metrics = _case_metrics(case)
        assignment_count = max(int(metrics.get("assignment_count", 0)), 1)
        ratios.append(
            int(metrics.get("duplicate_dispatch_assignment_count", 0))
            / assignment_count
        )
    return max(ratios) if ratios else 0.0


def _has_stagnation(history_summary: dict[str, Any]) -> bool:
    return any(
        bool(history_summary.get(key, False))
        for key in (
            "beam_staged_diminishing_returns",
            "portfolio_overlay_diminishing_returns",
            "pressure_targeted_diminishing_returns",
            "bundle_merge_duplicate_diminishing_returns",
            "task_risk_duplicate_diminishing_returns",
            "risk_tier_duplicate_diminishing_returns",
            "duplicate_augment_diminishing_returns",
        )
    )


def _dominant_worst_cases(history_summary: dict[str, Any]) -> list[str]:
    case_id = str(history_summary.get("dominant_worst_case_id", ""))
    return [case_id] if case_id else []


def _latest_best_path(history_summary: dict[str, Any]) -> str:
    for key in (
        "latest_beam_staged_best_variant_path",
        "latest_portfolio_overlay_best_variant_path",
        "latest_pressure_targeted_best_variant_path",
        "latest_bundle_merge_duplicate_best_variant_path",
        "latest_task_risk_duplicate_best_variant_path",
        "latest_risk_tier_duplicate_best_variant_path",
        "latest_duplicate_augment_best_variant_path",
        "latest_local_improve_best_variant_path",
        "latest_best_variant_path",
    ):
        value = str(history_summary.get(key, ""))
        if value:
            return value
    return ""


def _fallback_low_complexity_family(history_summary: dict[str, Any]) -> str:
    if int(history_summary.get("local_improve_sweep_count", 0)) == 0:
        return "local_improve"
    return "duplicate_augment"


def _low_complexity_region() -> dict[str, Any]:
    return {
        "profile": "low_complexity",
        "max_extra_dispatches_max": 12,
        "max_couriers_per_assignment_max": 2,
        "time_budget_seconds_max": 8.5,
    }


def _risk_duplicate_region(strategy_family: str) -> dict[str, Any]:
    if strategy_family == "low_willingness_deep_duplicate":
        return {
            "profile": "low_willingness_deep",
            "strategy_family": strategy_family,
            "high_risk_target_min": 0.95,
            "high_risk_target_max": 0.97,
            "max_extra_dispatches_min": 40,
            "max_extra_dispatches_max": 60,
            "max_couriers_per_assignment_min": 3,
            "max_couriers_per_assignment_max": 4,
        }
    return {
        "profile": "risk_focused",
        "strategy_family": strategy_family,
        "high_risk_target_min": 0.92,
        "high_risk_target_max": 0.95,
        "max_extra_dispatches_min": 20,
        "max_extra_dispatches_max": 35,
        "max_couriers_per_assignment_min": 2,
        "max_couriers_per_assignment_max": 3,
    }


def _bundle_merge_region() -> dict[str, Any]:
    return {
        "profile": "bundle_merge_focused",
        "merge_min_improvement_values": [0.0, -10.0],
        "max_extra_dispatches_min": 20,
        "max_extra_dispatches_max": 30,
        "max_couriers_per_assignment_min": 2,
        "max_couriers_per_assignment_max": 3,
    }


def _conservative_duplicate_region() -> dict[str, Any]:
    return {
        "profile": "conservative_duplicate",
        "max_extra_dispatches_max": 12,
        "max_couriers_per_assignment_max": 2,
        "min_roi_min": 0.0,
    }


def _local_improve_region() -> dict[str, Any]:
    return {
        "profile": "score_cost_local_improve",
        "alpha_min": 85.0,
        "alpha_max": 95.0,
        "max_passes_max": 2,
    }


def _unique(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
