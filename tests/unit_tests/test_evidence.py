from autoresearch_agent.research.evidence import build_evidence_profile


def test_evidence_low_expected_success_targets_risk_family() -> None:
    profile = build_evidence_profile(
        [
            _record(
                mean_expected_success_ratio=0.83,
                worst_case_id="low_willingness_stress_20_tasks",
                case_results=[
                    _case("low_willingness_stress_20_tasks", success=0.76)
                ],
            )
        ],
        {"risk_tier_duplicate_sweep_count": 0},
    )

    assert profile.failure_mode == "low_expected_success"
    assert profile.strategy_family == "risk_tier_duplicate"
    assert profile.param_region["high_risk_target_min"] == 0.92


def test_evidence_low_expected_success_after_risk_tier_targets_deep_duplicate() -> None:
    profile = build_evidence_profile(
        [
            _record(
                mean_expected_success_ratio=0.86,
                worst_case_id="low_willingness_stress_20_tasks",
                case_results=[
                    _case("low_willingness_stress_20_tasks", success=0.79)
                ],
            )
        ],
        {"risk_tier_duplicate_sweep_count": 1},
    )

    assert profile.failure_mode == "low_expected_success"
    assert profile.strategy_family == "low_willingness_deep_duplicate"
    assert profile.param_region["max_extra_dispatches_min"] == 40


def test_evidence_scarce_case_targets_bundle_merge() -> None:
    profile = build_evidence_profile(
        [
            _record(
                mean_expected_success_ratio=0.94,
                worst_case_id="scarce_couriers_40_tasks",
                case_results=[
                    _case(
                        "scarce_couriers_40_tasks",
                        success=0.91,
                        courier_usage=1.0,
                    )
                ],
            )
        ],
        {},
    )

    assert profile.failure_mode == "scarce_courier_pressure"
    assert profile.strategy_family == "bundle_merge_duplicate"


def test_evidence_invalid_or_timeout_uses_conservative_region() -> None:
    profile = build_evidence_profile(
        [
            _record(
                invalid_case_count=1,
                timeout_count=1,
                case_results=[_case("full", valid=False, timed_out=True)],
            )
        ],
        {},
    )

    assert profile.failure_mode == "invalid_or_timeout"
    assert profile.param_region["max_extra_dispatches_max"] == 12
    assert profile.param_region["max_couriers_per_assignment_max"] == 2


def test_evidence_stagnation_stops_when_no_failure_remains() -> None:
    profile = build_evidence_profile(
        [
            _record(best_path="best.py"),
            _record(best_path="best.py"),
        ],
        {
            "duplicate_augment_diminishing_returns": True,
            "latest_duplicate_augment_best_variant_path": "best.py",
        },
    )

    assert profile.failure_mode == "stagnation"
    assert profile.strategy_family == "stop_search"
    assert profile.param_region["recommended_solver_path"] == "best.py"


def _record(
    *,
    best_path: str = "best.py",
    mean_expected_success_ratio: float = 0.94,
    worst_case_id: str = "full",
    invalid_case_count: int = 0,
    timeout_count: int = 0,
    case_results: list[dict] | None = None,
) -> dict:
    return {
        "best_variant_path": best_path,
        "batch": {
            "variant_results": [
                {
                    "rank": 1,
                    "variant_path": best_path,
                    "case_results": case_results or [],
                    "aggregate_metrics": {
                        "is_valid": invalid_case_count == 0 and timeout_count == 0,
                        "invalid_case_count": invalid_case_count,
                        "timeout_count": timeout_count,
                        "mean_expected_success_ratio": mean_expected_success_ratio,
                        "mean_proxy_score": -100.0,
                        "worst_case_id": worst_case_id,
                    },
                }
            ]
        },
    }


def _case(
    case_id: str,
    *,
    success: float = 0.94,
    courier_usage: float = 0.5,
    valid: bool = True,
    timed_out: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "timed_out": timed_out,
        "metrics": {
            "is_valid": valid,
            "expected_success_ratio": success,
            "courier_usage_ratio": courier_usage,
            "bundled_task_assignment_count": 1,
            "duplicate_dispatch_assignment_count": 0,
            "assignment_count": 20,
            "total_score": 300.0,
            "score_per_expected_successful_task": 15.0,
        },
    }
