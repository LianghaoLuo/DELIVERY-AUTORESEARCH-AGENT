from autoresearch_agent.research.autopilot import (
    decide_next_experiment,
    summarize_local_history,
)


def test_empty_history_selects_broad_strategy() -> None:
    summary = summarize_local_history([])
    decision = decide_next_experiment(summary)

    assert decision.search_space == "broad_strategy"
    assert decision.failure_mode == "bootstrap"
    assert "baselines" in decision.hypothesis


def test_low_expected_success_selects_risk_tier_duplicate() -> None:
    summary = summarize_local_history(
        [
            _variant_record(
                "broad-strategy-sweep",
                mean_expected_success_ratio=0.82,
                worst_case_id="low_willingness_stress_20_tasks",
                case_results=[
                    _case_result(
                        "low_willingness_stress_20_tasks",
                        expected_success_ratio=0.78,
                    )
                ],
            )
        ]
    )
    decision = decide_next_experiment(summary)

    assert summary["evidence_profile"]["failure_mode"] == "low_expected_success"
    assert decision.search_space == "risk_tier_duplicate"
    assert decision.param_region["high_risk_target_min"] == 0.92
    assert decision.param_region["max_couriers_per_assignment_max"] == 3


def test_low_expected_success_after_risk_tier_selects_deep_duplicate() -> None:
    summary = summarize_local_history(
        [
            _variant_record("broad-strategy-sweep"),
            _variant_record(
                "risk-tier-duplicate-strategy-sweep",
                best_path="risk.py",
                mean_expected_success_ratio=0.86,
                worst_case_id="low_willingness_20_tasks",
                case_results=[
                    _case_result(
                        "low_willingness_20_tasks",
                        expected_success_ratio=0.81,
                    )
                ],
            ),
        ]
    )
    decision = decide_next_experiment(summary)

    assert decision.failure_mode == "low_expected_success"
    assert decision.search_space == "low_willingness_deep_duplicate"
    assert decision.param_region["max_extra_dispatches_min"] == 40


def test_scarce_courier_pressure_selects_bundle_merge() -> None:
    summary = summarize_local_history(
        [
            _variant_record(
                "broad-strategy-sweep",
                mean_expected_success_ratio=0.93,
                worst_case_id="scarce_couriers_40_tasks",
                case_results=[
                    _case_result(
                        "scarce_couriers_40_tasks",
                        expected_success_ratio=0.91,
                        courier_usage_ratio=1.0,
                    )
                ],
            )
        ]
    )
    decision = decide_next_experiment(summary)

    assert decision.failure_mode == "scarce_courier_pressure"
    assert decision.search_space == "bundle_merge_duplicate"
    assert decision.param_region["merge_min_improvement_values"] == [0.0, -10.0]


def test_repeated_scarce_bundle_merge_advances_to_pressure_targeted() -> None:
    records = [
        _variant_record("broad-strategy-sweep", best_path="broad.py"),
        _variant_record(
            "bundle-merge-duplicate-strategy-sweep",
            best_path="bundle.py",
            second_path="bundle_b.py",
            mean_expected_success_ratio=0.93,
            worst_case_id="scarce_couriers_40_tasks",
            case_results=[
                _case_result(
                    "scarce_couriers_40_tasks",
                    expected_success_ratio=0.91,
                    courier_usage_ratio=1.0,
                )
            ],
        ),
        _variant_record(
            "bundle-merge-duplicate-strategy-sweep",
            best_path="bundle.py",
            second_path="bundle_b.py",
            mean_expected_success_ratio=0.93,
            worst_case_id="scarce_couriers_40_tasks",
            case_results=[
                _case_result(
                    "scarce_couriers_40_tasks",
                    expected_success_ratio=0.91,
                    courier_usage_ratio=1.0,
                )
            ],
        ),
    ]

    summary = summarize_local_history(records)
    decision = decide_next_experiment(summary)

    assert summary["bundle_merge_duplicate_diminishing_returns"]
    assert decision.search_space == "pressure_targeted"
    assert decision.failure_mode == "scarce_courier_pressure"
    assert decision.strategy_family == "pressure_targeted"
    assert decision.param_region["previous_search_space"] == "bundle_merge_duplicate"


def test_invalid_or_timeout_selects_low_complexity_region() -> None:
    summary = summarize_local_history(
        [
            _variant_record(
                "broad-strategy-sweep",
                invalid_case_count=1,
                timeout_count=1,
                case_results=[
                    _case_result(
                        "full",
                        is_valid=False,
                        timed_out=True,
                    )
                ],
            )
        ]
    )
    decision = decide_next_experiment(summary)

    assert decision.failure_mode == "invalid_or_timeout"
    assert decision.search_space == "local_improve"
    assert decision.param_region["profile"] == "low_complexity"
    assert decision.param_region["max_couriers_per_assignment_max"] == 2


def test_repeated_low_willingness_deep_advances_to_pressure_targeted() -> None:
    records = [
        _variant_record("broad-strategy-sweep"),
        _variant_record("risk-tier-duplicate-strategy-sweep", best_path="risk.py"),
        _variant_record(
            "low-willingness-deep-duplicate-strategy-sweep",
            best_path="deep.py",
            second_path="deep_b.py",
            mean_expected_success_ratio=0.86,
            worst_case_id="low_willingness_20_tasks",
            case_results=[
                _case_result(
                    "low_willingness_20_tasks",
                    expected_success_ratio=0.81,
                )
            ],
        ),
        _variant_record(
            "low-willingness-deep-duplicate-strategy-sweep",
            best_path="deep.py",
            second_path="deep_b.py",
            mean_expected_success_ratio=0.86,
            worst_case_id="low_willingness_20_tasks",
            case_results=[
                _case_result(
                    "low_willingness_20_tasks",
                    expected_success_ratio=0.81,
                )
            ],
        ),
    ]

    summary = summarize_local_history(records)
    decision = decide_next_experiment(summary)

    assert summary["low_willingness_deep_duplicate_diminishing_returns"]
    assert decision.search_space == "pressure_targeted"
    assert decision.failure_mode == "low_expected_success"
    assert decision.param_region["previous_search_space"] == (
        "low_willingness_deep_duplicate"
    )


def test_repeated_pressure_targeted_advances_to_portfolio_overlay() -> None:
    records = [
        _variant_record("broad-strategy-sweep", best_path="broad.py"),
        *_scarce_diminishing_records(
            "bundle-merge-duplicate-strategy-sweep",
            best_path="bundle.py",
        ),
        *_scarce_diminishing_records(
            "pressure-targeted-strategy-sweep",
            best_path="pressure.py",
        ),
    ]

    summary = summarize_local_history(records)
    decision = decide_next_experiment(summary)

    assert summary["pressure_targeted_diminishing_returns"]
    assert decision.search_space == "portfolio_overlay"
    assert decision.param_region["previous_search_space"] == "bundle_merge_duplicate"


def test_repeated_portfolio_overlay_advances_to_beam_staged() -> None:
    records = [
        _variant_record("broad-strategy-sweep", best_path="broad.py"),
        *_scarce_diminishing_records(
            "bundle-merge-duplicate-strategy-sweep",
            best_path="bundle.py",
        ),
        *_scarce_diminishing_records(
            "pressure-targeted-strategy-sweep",
            best_path="pressure.py",
        ),
        *_scarce_diminishing_records(
            "portfolio-overlay-strategy-sweep",
            best_path="portfolio.py",
            second_proxy=-50.0,
        ),
    ]

    summary = summarize_local_history(records)
    decision = decide_next_experiment(summary)

    assert summary["portfolio_overlay_diminishing_returns"]
    assert decision.search_space == "beam_staged"


def test_repeated_beam_staged_stops_with_local_evidence() -> None:
    records = [
        _variant_record("broad-strategy-sweep", best_path="broad.py"),
        *_scarce_diminishing_records(
            "bundle-merge-duplicate-strategy-sweep",
            best_path="bundle.py",
        ),
        *_scarce_diminishing_records(
            "pressure-targeted-strategy-sweep",
            best_path="pressure.py",
        ),
        *_scarce_diminishing_records(
            "portfolio-overlay-strategy-sweep",
            best_path="portfolio.py",
        ),
        *_scarce_diminishing_records(
            "beam-staged-strategy-sweep",
            best_path="beam.py",
        ),
    ]

    summary = summarize_local_history(records)
    decision = decide_next_experiment(summary)

    assert summary["beam_staged_diminishing_returns"]
    assert decision.search_space == "stop_search"
    assert decision.recommended_solver_path == "beam.py"
    assert decision.param_region["stalled_search_space"] == "bundle_merge_duplicate"


def test_stable_history_without_failure_stops_search() -> None:
    records = [
        _variant_record("broad-strategy-sweep", best_path="broad.py"),
        _variant_record("duplicate-augment-strategy-sweep", best_path="best.py"),
        _variant_record("duplicate-augment-strategy-sweep", best_path="best.py"),
    ]

    summary = summarize_local_history(records)
    decision = decide_next_experiment(summary)

    assert summary["duplicate_augment_diminishing_returns"]
    assert decision.failure_mode == "stagnation"
    assert decision.search_space == "stop_search"
    assert decision.recommended_solver_path == "best.py"


def test_legacy_numbered_labels_still_count_as_strategy_stages() -> None:
    records = [
        _variant_record("p1-strategy-search-sweep"),
        _variant_record("p1c-local-strategy-search-sweep", best_path="best.py"),
        _variant_record("p1c-local-strategy-search-sweep", best_path="best.py"),
    ]

    summary = summarize_local_history(records)

    assert summary["broad_strategy_sweep_count"] == 1
    assert summary["local_improve_sweep_count"] == 2


def _variant_record(
    label: str,
    *,
    best_path: str = "variant_a.py",
    second_path: str = "variant_b.py",
    best_proxy: float = -100.0,
    second_proxy: float = -95.0,
    invalid_case_count: int = 0,
    timeout_count: int = 0,
    mean_expected_success_ratio: float = 0.94,
    worst_case_id: str = "full",
    case_results: list[dict] | None = None,
) -> dict:
    return {
        "label": label,
        "best_variant_path": best_path,
        "batch": {
            "variant_results": [
                _variant_result(
                    1,
                    best_path,
                    best_proxy,
                    invalid_case_count=invalid_case_count,
                    timeout_count=timeout_count,
                    mean_expected_success_ratio=mean_expected_success_ratio,
                    worst_case_id=worst_case_id,
                    case_results=case_results,
                ),
                _variant_result(
                    2,
                    second_path,
                    second_proxy,
                    mean_expected_success_ratio=mean_expected_success_ratio,
                    worst_case_id=worst_case_id,
                    case_results=case_results,
                ),
            ]
        },
    }


def _scarce_diminishing_records(
    label: str,
    *,
    best_path: str,
    second_proxy: float = -95.0,
) -> list[dict]:
    return [
        _variant_record(
            label,
            best_path=best_path,
            second_path=f"{best_path}.b",
            second_proxy=second_proxy,
            mean_expected_success_ratio=0.93,
            worst_case_id="scarce_couriers_40_tasks",
            case_results=[
                _case_result(
                    "scarce_couriers_40_tasks",
                    expected_success_ratio=0.91,
                    courier_usage_ratio=1.0,
                )
            ],
        ),
        _variant_record(
            label,
            best_path=best_path,
            second_path=f"{best_path}.b",
            second_proxy=second_proxy,
            mean_expected_success_ratio=0.93,
            worst_case_id="scarce_couriers_40_tasks",
            case_results=[
                _case_result(
                    "scarce_couriers_40_tasks",
                    expected_success_ratio=0.91,
                    courier_usage_ratio=1.0,
                )
            ],
        ),
    ]


def _variant_result(
    rank: int,
    path: str,
    mean_proxy_score: float,
    *,
    invalid_case_count: int = 0,
    timeout_count: int = 0,
    mean_expected_success_ratio: float = 0.94,
    worst_case_id: str = "full",
    case_results: list[dict] | None = None,
) -> dict:
    return {
        "rank": rank,
        "variant_path": path,
        "case_results": case_results or [],
        "aggregate_metrics": {
            "is_valid": invalid_case_count == 0 and timeout_count == 0,
            "case_count": 5,
            "timeout_count": timeout_count,
            "invalid_case_count": invalid_case_count,
            "mean_proxy_score": mean_proxy_score,
            "mean_expected_success_ratio": mean_expected_success_ratio,
            "mean_task_coverage_ratio": 1.0,
            "worst_case_id": worst_case_id,
        },
    }


def _case_result(
    case_id: str,
    *,
    expected_success_ratio: float = 0.94,
    courier_usage_ratio: float = 0.5,
    bundled_task_assignment_count: int = 1,
    duplicate_dispatch_assignment_count: int = 0,
    assignment_count: int = 20,
    total_score: float = 300.0,
    is_valid: bool = True,
    timed_out: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "timed_out": timed_out,
        "metrics": {
            "is_valid": is_valid,
            "expected_success_ratio": expected_success_ratio,
            "courier_usage_ratio": courier_usage_ratio,
            "bundled_task_assignment_count": bundled_task_assignment_count,
            "duplicate_dispatch_assignment_count": duplicate_dispatch_assignment_count,
            "assignment_count": assignment_count,
            "total_score": total_score,
            "score_per_expected_successful_task": total_score / assignment_count,
        },
    }
