"""Local-history-driven AutoResearch strategy planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from autoresearch_agent.research.evidence import (
    build_evidence_profile,
    evidence_profile_to_dict,
)
from autoresearch_agent.research.strategy_space import strategy_sweep_label

SearchSpace = Literal[
    "fixed_variants",
    "broad_strategy",
    "local_improve",
    "duplicate_augment",
    "duplicate_augment_refine",
    "risk_tier_duplicate",
    "low_willingness_deep_duplicate",
    "task_risk_duplicate",
    "bundle_split_duplicate",
    "bundle_merge_duplicate",
    "pressure_targeted",
    "portfolio_overlay",
    "beam_staged",
    "stop_search",
]

BROAD_STRATEGY_LABEL = strategy_sweep_label("broad_strategy")
LOCAL_IMPROVE_LABEL = strategy_sweep_label("local_improve")
DUPLICATE_AUGMENT_LABEL = strategy_sweep_label("duplicate_augment")
DUPLICATE_AUGMENT_REFINE_LABEL = strategy_sweep_label("duplicate_augment_refine")
RISK_TIER_DUPLICATE_LABEL = strategy_sweep_label("risk_tier_duplicate")
LOW_WILLINGNESS_DEEP_DUPLICATE_LABEL = strategy_sweep_label(
    "low_willingness_deep_duplicate"
)
TASK_RISK_DUPLICATE_LABEL = strategy_sweep_label("task_risk_duplicate")
BUNDLE_SPLIT_DUPLICATE_LABEL = strategy_sweep_label("bundle_split_duplicate")
BUNDLE_MERGE_DUPLICATE_LABEL = strategy_sweep_label("bundle_merge_duplicate")
PRESSURE_TARGETED_LABEL = strategy_sweep_label("pressure_targeted")
PORTFOLIO_OVERLAY_LABEL = strategy_sweep_label("portfolio_overlay")
BEAM_STAGED_LABEL = strategy_sweep_label("beam_staged")
FIXED_LABEL = "graph-fixed-variant-suite-batch"
LEGACY_P1_LABEL = "p1-strategy-search-sweep"
LEGACY_P1C_LABEL = "p1c-local-strategy-search-sweep"


@dataclass(frozen=True)
class ResearchDecision:
    """One deterministic next-step decision for the local research loop."""

    search_space: SearchSpace
    hypothesis: str
    reasons: list[str]
    expected_evidence: list[str]
    output_dir: str
    candidate_limit: int = 4
    recommended_solver_path: str = ""
    failure_mode: str = ""
    strategy_family: str = ""
    param_region: dict[str, Any] | None = None
    selected_config_ids: list[str] | None = None
    configs: list[dict[str, Any]] | None = None
    evidence_score: float = 0.0


def summarize_local_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize append-only local experiment records for graph decisions."""
    variant_records = [record for record in records if _is_variant_suite_record(record)]
    broad_records = [
        record for record in variant_records if _record_label(record) in _broad_labels()
    ]
    local_improve_records = [
        record
        for record in variant_records
        if _record_label(record) in _local_improve_labels()
    ]
    duplicate_records = [
        record
        for record in variant_records
        if _record_label(record) == DUPLICATE_AUGMENT_LABEL
    ]
    duplicate_refine_records = [
        record
        for record in variant_records
        if _record_label(record) == DUPLICATE_AUGMENT_REFINE_LABEL
    ]
    risk_tier_records = [
        record
        for record in variant_records
        if _record_label(record) == RISK_TIER_DUPLICATE_LABEL
    ]
    low_willingness_deep_records = [
        record
        for record in variant_records
        if _record_label(record) == LOW_WILLINGNESS_DEEP_DUPLICATE_LABEL
    ]
    task_risk_records = [
        record
        for record in variant_records
        if _record_label(record) == TASK_RISK_DUPLICATE_LABEL
    ]
    bundle_split_records = [
        record
        for record in variant_records
        if _record_label(record) == BUNDLE_SPLIT_DUPLICATE_LABEL
    ]
    bundle_merge_records = [
        record
        for record in variant_records
        if _record_label(record) == BUNDLE_MERGE_DUPLICATE_LABEL
    ]
    pressure_targeted_records = [
        record
        for record in variant_records
        if _record_label(record) == PRESSURE_TARGETED_LABEL
    ]
    portfolio_overlay_records = [
        record
        for record in variant_records
        if _record_label(record) == PORTFOLIO_OVERLAY_LABEL
    ]
    beam_staged_records = [
        record
        for record in variant_records
        if _record_label(record) == BEAM_STAGED_LABEL
    ]
    fixed_records = [
        record for record in variant_records if _record_label(record) == FIXED_LABEL
    ]
    latest_variant = variant_records[-1] if variant_records else {}
    latest_broad = broad_records[-1] if broad_records else {}
    latest_local_improve = local_improve_records[-1] if local_improve_records else {}
    latest_duplicate = duplicate_records[-1] if duplicate_records else {}
    latest_duplicate_refine = (
        duplicate_refine_records[-1] if duplicate_refine_records else {}
    )
    latest_risk_tier = risk_tier_records[-1] if risk_tier_records else {}
    latest_low_willingness_deep = (
        low_willingness_deep_records[-1] if low_willingness_deep_records else {}
    )
    latest_task_risk = task_risk_records[-1] if task_risk_records else {}
    latest_bundle_split = bundle_split_records[-1] if bundle_split_records else {}
    latest_bundle_merge = bundle_merge_records[-1] if bundle_merge_records else {}
    latest_pressure_targeted = (
        pressure_targeted_records[-1] if pressure_targeted_records else {}
    )
    latest_portfolio_overlay = (
        portfolio_overlay_records[-1] if portfolio_overlay_records else {}
    )
    latest_beam_staged = beam_staged_records[-1] if beam_staged_records else {}
    latest_strategy_records = [
        record
        for record in variant_records
        if _record_label(record) in _strategy_labels()
    ]
    latest_sweep = (
        latest_strategy_records[-1] if latest_strategy_records else latest_variant
    )
    local_improve_diminishing = _strategy_is_diminishing(local_improve_records)
    duplicate_diminishing = _strategy_is_diminishing(duplicate_records)
    risk_tier_diminishing = _strategy_is_diminishing(risk_tier_records)
    low_willingness_deep_diminishing = _strategy_is_diminishing(
        low_willingness_deep_records
    )
    task_risk_diminishing = _strategy_is_diminishing(task_risk_records)
    bundle_merge_diminishing = _strategy_is_diminishing(bundle_merge_records)
    pressure_targeted_diminishing = _strategy_is_diminishing(
        pressure_targeted_records
    )
    portfolio_overlay_diminishing = _strategy_is_diminishing(
        portfolio_overlay_records
    )
    beam_staged_diminishing = _strategy_is_diminishing(beam_staged_records)
    worst_case_counts = _worst_case_counts(variant_records)
    summary = {
        "record_count": len(records),
        "variant_suite_record_count": len(variant_records),
        "fixed_variant_run_count": len(fixed_records),
        "broad_strategy_sweep_count": len(broad_records),
        "local_improve_sweep_count": len(local_improve_records),
        "duplicate_augment_sweep_count": len(duplicate_records),
        "duplicate_augment_refine_sweep_count": len(duplicate_refine_records),
        "risk_tier_duplicate_sweep_count": len(risk_tier_records),
        "low_willingness_deep_duplicate_sweep_count": len(
            low_willingness_deep_records
        ),
        "task_risk_duplicate_sweep_count": len(task_risk_records),
        "bundle_split_duplicate_sweep_count": len(bundle_split_records),
        "bundle_merge_duplicate_sweep_count": len(bundle_merge_records),
        "pressure_targeted_sweep_count": len(pressure_targeted_records),
        "portfolio_overlay_sweep_count": len(portfolio_overlay_records),
        "beam_staged_sweep_count": len(beam_staged_records),
        "latest_variant_label": _record_label(latest_variant),
        "latest_sweep_label": _record_label(latest_sweep),
        "latest_best_variant_path": str(latest_sweep.get("best_variant_path", "")),
        "latest_broad_strategy_best_variant_path": str(
            latest_broad.get("best_variant_path", "")
        ),
        "latest_local_improve_best_variant_path": str(
            latest_local_improve.get("best_variant_path", "")
        ),
        "latest_duplicate_augment_best_variant_path": str(
            latest_duplicate.get("best_variant_path", "")
        ),
        "latest_duplicate_augment_refine_best_variant_path": str(
            latest_duplicate_refine.get("best_variant_path", "")
        ),
        "latest_risk_tier_duplicate_best_variant_path": str(
            latest_risk_tier.get("best_variant_path", "")
        ),
        "latest_low_willingness_deep_duplicate_best_variant_path": str(
            latest_low_willingness_deep.get("best_variant_path", "")
        ),
        "latest_task_risk_duplicate_best_variant_path": str(
            latest_task_risk.get("best_variant_path", "")
        ),
        "latest_bundle_split_duplicate_best_variant_path": str(
            latest_bundle_split.get("best_variant_path", "")
        ),
        "latest_bundle_merge_duplicate_best_variant_path": str(
            latest_bundle_merge.get("best_variant_path", "")
        ),
        "latest_pressure_targeted_best_variant_path": str(
            latest_pressure_targeted.get("best_variant_path", "")
        ),
        "latest_portfolio_overlay_best_variant_path": str(
            latest_portfolio_overlay.get("best_variant_path", "")
        ),
        "latest_beam_staged_best_variant_path": str(
            latest_beam_staged.get("best_variant_path", "")
        ),
        "latest_top_proxy_gap": _top_proxy_gap(latest_sweep),
        "latest_has_invalid_or_timeout": _has_invalid_or_timeout(latest_sweep),
        "recent_local_improve_best_repetition_count": _recent_best_repetition_count(
            local_improve_records
        ),
        "recent_duplicate_augment_best_repetition_count": _recent_best_repetition_count(
            duplicate_records
        ),
        "recent_local_improve_top_proxy_gaps": _recent_top_proxy_gaps(
            local_improve_records
        ),
        "recent_duplicate_augment_top_proxy_gaps": _recent_top_proxy_gaps(
            duplicate_records
        ),
        "local_improve_diminishing_returns": local_improve_diminishing,
        "duplicate_augment_diminishing_returns": duplicate_diminishing,
        "risk_tier_duplicate_diminishing_returns": risk_tier_diminishing,
        "low_willingness_deep_duplicate_diminishing_returns": (
            low_willingness_deep_diminishing
        ),
        "task_risk_duplicate_diminishing_returns": task_risk_diminishing,
        "bundle_merge_duplicate_diminishing_returns": bundle_merge_diminishing,
        "pressure_targeted_diminishing_returns": pressure_targeted_diminishing,
        "portfolio_overlay_diminishing_returns": portfolio_overlay_diminishing,
        "beam_staged_diminishing_returns": beam_staged_diminishing,
        "worst_case_counts": worst_case_counts,
        "dominant_worst_case_id": _dominant_key(worst_case_counts),
    }
    summary["evidence_profile"] = evidence_profile_to_dict(
        build_evidence_profile(records, summary)
    )
    return summary


def decide_next_experiment(
    history_summary: dict[str, Any],
) -> ResearchDecision:
    """Choose the next local experiment from local failure-mode evidence."""
    if int(history_summary.get("variant_suite_record_count", 0)) == 0:
        return ResearchDecision(
            search_space="broad_strategy",
            hypothesis=(
                "Need to establish score/willingness/bundle baselines before "
                "narrow local refinement."
            ),
            reasons=[
                "No local variant-suite history is available.",
                "A broad strategy sweep creates the first comparable local memory.",
            ],
            expected_evidence=[
                "Rank score-alpha, failure-penalty, bundle-bias, and local-improve families.",
                "Identify whether probability-aware ordering beats score-only greedy locally.",
            ],
            output_dir="experiments/generated_variants/broad_strategy",
            failure_mode="bootstrap",
            strategy_family="broad_strategy",
            param_region={"profile": "broad"},
            evidence_score=1.0,
        )

    if int(history_summary.get("broad_strategy_sweep_count", 0)) == 0:
        return ResearchDecision(
            search_space="broad_strategy",
            hypothesis=(
                "Need a broad local strategy sweep to compare family-level "
                "tradeoffs before deeper refinement."
            ),
            reasons=[
                "Local history exists, but no broad strategy sweep was found.",
                "Fixed variants alone do not expose enough strategy-family evidence.",
            ],
            expected_evidence=[
                "Compare greedy_sort, parameter_search, and local_improve families.",
                "Use robust-suite proxy metrics to select a refinement neighborhood.",
            ],
            output_dir="experiments/generated_variants/broad_strategy",
            failure_mode="bootstrap",
            strategy_family="broad_strategy",
            param_region={"profile": "broad"},
            evidence_score=1.0,
        )

    profile = _profile_from_summary(history_summary)
    search_space = _search_space_for_profile(profile)
    if _selected_search_space_is_diminishing(history_summary, search_space):
        advanced_decision = _advanced_probe_decision(
            history_summary,
            profile,
            search_space,
        )
        if advanced_decision is not None:
            return advanced_decision
        pressure_sequence_complete = _advanced_probe_sequence_is_complete(
            history_summary,
            profile,
        )
        recommended_solver_path = _latest_recommended_solver(history_summary)
        return ResearchDecision(
            search_space="stop_search",
            hypothesis=(
                "The v3 pressure probe sequence has settled locally; stop the "
                "local loop and preserve the current best candidate as evidence."
                if pressure_sequence_complete
                else (
                    f"`{search_space}` has repeated the same local best; stop "
                    "the local loop and preserve the current best candidate as "
                    "evidence."
                )
            ),
            reasons=[
                *[str(reason) for reason in profile.get("reasons", [])],
                (
                    "All v3 pressure probe stages have repeated stable local "
                    "best candidates."
                    if pressure_sequence_complete
                    else f"Selected local family `{search_space}` has diminishing returns."
                ),
                "External benchmark score records are not used for this stop decision.",
            ],
            expected_evidence=[
                "Re-run the recommended solver as stop-search evidence.",
                "Confirm validity, timeout safety, and repeated local ranking.",
            ],
            output_dir="",
            recommended_solver_path=recommended_solver_path,
            failure_mode="stagnation",
            strategy_family="stop_search",
            param_region={
                "recommended_solver_path": recommended_solver_path,
                "stalled_search_space": search_space,
                "settled_v3_pressure_sequence": pressure_sequence_complete,
            },
            evidence_score=float(profile.get("evidence_score", 0.2)),
        )
    recommended_solver_path = ""
    output_dir = _output_dir_for_search_space(search_space)
    if search_space == "stop_search":
        recommended_solver_path = str(
            profile.get("param_region", {}).get("recommended_solver_path", "")
        ) or _latest_recommended_solver(history_summary)
        output_dir = ""
    return ResearchDecision(
        search_space=search_space,
        hypothesis=_hypothesis_for_profile(profile, search_space),
        reasons=[
            *[str(reason) for reason in profile.get("reasons", [])],
            (
                "Affected cases: "
                f"{', '.join(profile.get('affected_cases', []) or ['none'])}."
            ),
            f"Selected parameter region: {profile.get('param_region', {})}.",
        ],
        expected_evidence=_expected_evidence_for_profile(profile, search_space),
        output_dir=output_dir,
        recommended_solver_path=recommended_solver_path,
        failure_mode=str(profile.get("failure_mode", "")),
        strategy_family=str(profile.get("strategy_family", search_space)),
        param_region=dict(profile.get("param_region", {})),
        evidence_score=float(profile.get("evidence_score", 0.0)),
    )


def research_decision_to_dict(decision: ResearchDecision) -> dict[str, Any]:
    """Convert a research decision to JSON-serializable primitives."""
    return {
        "search_space": decision.search_space,
        "hypothesis": decision.hypothesis,
        "reasons": list(decision.reasons),
        "expected_evidence": list(decision.expected_evidence),
        "output_dir": decision.output_dir,
        "candidate_limit": decision.candidate_limit,
        "recommended_solver_path": decision.recommended_solver_path,
        "failure_mode": decision.failure_mode,
        "strategy_family": decision.strategy_family,
        "param_region": dict(decision.param_region or {}),
        "config_ids": list(decision.selected_config_ids or []),
        "configs": list(decision.configs or []),
        "evidence_score": decision.evidence_score,
    }


def _profile_from_summary(history_summary: dict[str, Any]) -> dict[str, Any]:
    profile = history_summary.get("evidence_profile", {})
    return profile if isinstance(profile, dict) else {}


def _search_space_for_profile(profile: dict[str, Any]) -> SearchSpace:
    strategy_family = str(profile.get("strategy_family", "local_improve"))
    if strategy_family in {
        "broad_strategy",
        "local_improve",
        "duplicate_augment",
        "risk_tier_duplicate",
        "low_willingness_deep_duplicate",
        "task_risk_duplicate",
        "bundle_split_duplicate",
        "bundle_merge_duplicate",
        "pressure_targeted",
        "portfolio_overlay",
        "beam_staged",
        "stop_search",
    }:
        return strategy_family  # type: ignore[return-value]
    return "local_improve"


def _output_dir_for_search_space(search_space: SearchSpace) -> str:
    if search_space == "stop_search":
        return ""
    return f"experiments/generated_variants/{search_space}"


def _hypothesis_for_profile(profile: dict[str, Any], search_space: SearchSpace) -> str:
    failure_mode = str(profile.get("failure_mode", "unknown"))
    affected_cases = profile.get("affected_cases", [])
    affected = ", ".join(str(case) for case in affected_cases) if affected_cases else ""
    if search_space == "stop_search":
        return (
            "Local evidence has stabilized without a stronger remaining failure "
            "mode, so re-run the current best candidate as stop evidence."
        )
    return (
        f"Observed failure mode `{failure_mode}`"
        f"{f' on {affected}' if affected else ''}; run `{search_space}` in the "
        "selected parameter region to gather targeted evidence."
    )


def _expected_evidence_for_profile(
    profile: dict[str, Any],
    search_space: SearchSpace,
) -> list[str]:
    failure_mode = str(profile.get("failure_mode", "unknown"))
    if search_space == "stop_search":
        return [
            "Re-run the recommended solver as stop-search evidence.",
            "Confirm validity, timeout safety, and repeated local ranking.",
        ]
    if failure_mode == "invalid_or_timeout":
        return [
            "Confirm the conservative region removes invalid or timeout cases.",
            "Check whether mean proxy remains competitive after reducing complexity.",
        ]
    if failure_mode == "scarce_courier_pressure":
        return [
            "Measure courier usage and bundled coverage on scarce-courier cases.",
            "Check whether bundle merge improves worst-case proxy without timeout risk.",
        ]
    if failure_mode == "low_expected_success":
        return [
            "Measure expected-success movement on low-willingness stress cases.",
            "Inspect duplicate counts and score cost in the targeted risk region.",
        ]
    if failure_mode == "bundle_underuse":
        return [
            "Measure bundled assignment usage on bundle-dense cases.",
            "Check whether merge thresholds reduce courier pressure.",
        ]
    if failure_mode == "over_duplicate_cost":
        return [
            "Confirm conservative duplicate limits reduce score cost.",
            "Check expected-success regression against the previous local best.",
        ]
    return [
        "Measure mean proxy and worst-case movement in the selected region.",
        "Check validity, timeout risk, and output-signature stability.",
    ]


def _latest_recommended_solver(history_summary: dict[str, Any]) -> str:
    for key in (
        "latest_beam_staged_best_variant_path",
        "latest_portfolio_overlay_best_variant_path",
        "latest_pressure_targeted_best_variant_path",
        "latest_bundle_merge_duplicate_best_variant_path",
        "latest_task_risk_duplicate_best_variant_path",
        "latest_low_willingness_deep_duplicate_best_variant_path",
        "latest_risk_tier_duplicate_best_variant_path",
        "latest_duplicate_augment_best_variant_path",
        "latest_local_improve_best_variant_path",
        "latest_best_variant_path",
    ):
        value = str(history_summary.get(key, ""))
        if value:
            return value
    return ""


def _selected_search_space_is_diminishing(
    history_summary: dict[str, Any],
    search_space: SearchSpace,
) -> bool:
    if search_space == "stop_search":
        return False
    return bool(history_summary.get(f"{search_space}_diminishing_returns", False))


def _advanced_probe_decision(
    history_summary: dict[str, Any],
    profile: dict[str, Any],
    stalled_search_space: SearchSpace,
) -> ResearchDecision | None:
    """Move persistent pressure failures into the v3 structural probe sequence."""
    if str(profile.get("failure_mode", "")) not in {
        "scarce_courier_pressure",
        "low_expected_success",
        "bundle_underuse",
    }:
        return None

    stages: tuple[SearchSpace, ...] = (
        "pressure_targeted",
        "portfolio_overlay",
        "beam_staged",
    )
    start_index = (
        stages.index(stalled_search_space) + 1 if stalled_search_space in stages else 0
    )
    for search_space in stages[start_index:]:
        if not _advanced_stage_is_settled(history_summary, search_space):
            return ResearchDecision(
                search_space=search_space,
                hypothesis=(
                    f"`{stalled_search_space}` has converged locally while "
                    "pressure cases still dominate; run the next v3 structural "
                    f"probe `{search_space}`."
                ),
                reasons=[
                    *[str(reason) for reason in profile.get("reasons", [])],
                    (
                        f"Selected local family `{stalled_search_space}` has "
                        "diminishing returns."
                    ),
                    (
                        f"`{search_space}` has not yet produced settled local "
                        "evidence in this v3 run."
                    ),
                    "External benchmark score records are excluded from this loop decision.",
                ],
                expected_evidence=[
                    "Compare the structural probe against the current repeated local best.",
                    "Inspect pressure-case proxy movement, timeout risk, and output stability.",
                ],
                output_dir=_output_dir_for_search_space(search_space),
                failure_mode=str(profile.get("failure_mode", "stagnation")),
                strategy_family=search_space,
                param_region={
                    "profile": "v3_pressure_sequence",
                    "previous_search_space": stalled_search_space,
                },
                evidence_score=float(profile.get("evidence_score", 0.2)),
            )
    return None


def _advanced_stage_is_settled(
    history_summary: dict[str, Any],
    search_space: SearchSpace,
) -> bool:
    return int(history_summary.get(f"{search_space}_sweep_count", 0)) > 0 and bool(
        history_summary.get(f"{search_space}_diminishing_returns", False)
    )


def _advanced_probe_sequence_is_complete(
    history_summary: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    if str(profile.get("failure_mode", "")) not in {
        "scarce_courier_pressure",
        "low_expected_success",
        "bundle_underuse",
    }:
        return False
    stages: tuple[SearchSpace, ...] = (
        "pressure_targeted",
        "portfolio_overlay",
        "beam_staged",
    )
    return all(
        _advanced_stage_is_settled(history_summary, search_space)
        for search_space in stages
    )


def _continue_decision(
    *,
    search_space: SearchSpace,
    output_dir: str,
    hypothesis: str,
    reasons: list[str],
) -> ResearchDecision:
    """Return a repeat-run decision for an unsettled strategy space."""
    return ResearchDecision(
        search_space=search_space,
        hypothesis=hypothesis,
        reasons=reasons,
        expected_evidence=[
            "Confirm whether top variants repeat across runs.",
            "Inspect top-proxy gaps, worst-case concentration, and timeout risk.",
        ],
        output_dir=output_dir,
        failure_mode="stagnation",
        strategy_family=search_space,
        param_region={"profile": "continue"},
    )


def _is_variant_suite_record(record: dict[str, Any]) -> bool:
    return isinstance(record.get("batch"), dict) and isinstance(
        record.get("batch", {}).get("variant_results"), list
    )


def _record_label(record: dict[str, Any]) -> str:
    return str(record.get("label", ""))


def _record_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    batch = record.get("batch", {})
    results = batch.get("variant_results", []) if isinstance(batch, dict) else []
    return results if isinstance(results, list) else []


def _top_proxy_gap(record: dict[str, Any]) -> float:
    valid_scores = [
        float(result.get("aggregate_metrics", {}).get("mean_proxy_score", 0.0))
        for result in _record_results(record)
        if bool(result.get("aggregate_metrics", {}).get("is_valid", False))
    ]
    if len(valid_scores) < 2:
        return 0.0
    valid_scores = sorted(valid_scores)
    return valid_scores[1] - valid_scores[0]


def _has_invalid_or_timeout(record: dict[str, Any]) -> bool:
    for result in _record_results(record):
        aggregate = result.get("aggregate_metrics", {})
        if not bool(aggregate.get("is_valid", False)):
            return True
        if int(aggregate.get("timeout_count", 0)) > 0:
            return True
        if int(aggregate.get("invalid_case_count", 0)) > 0:
            return True
    return False


def _recent_best_repetition_count(records: list[dict[str, Any]], limit: int = 3) -> int:
    best_keys = [
        _best_identity(record) for record in records[-limit:] if _best_identity(record)
    ]
    if not best_keys:
        return 0
    return max(best_keys.count(key) for key in set(best_keys))


def _best_identity(record: dict[str, Any]) -> str:
    results = _record_results(record)
    best = next(
        (
            result
            for result in results
            if int(result.get("rank", 0)) == 1
            and bool(result.get("aggregate_metrics", {}).get("is_valid", False))
        ),
        results[0] if results else {},
    )
    signature = str(best.get("output_signature", ""))
    if signature:
        return signature
    return str(record.get("best_variant_path", ""))


def _recent_top_proxy_gaps(
    records: list[dict[str, Any]], limit: int = 3
) -> list[float]:
    return [_top_proxy_gap(record) for record in records[-limit:]]


def _strategy_is_diminishing(records: list[dict[str, Any]]) -> bool:
    if len(records) < 2:
        return False
    recent = records[-3:]
    if any(_has_invalid_or_timeout(record) for record in recent):
        return False
    gaps = _recent_top_proxy_gaps(records)
    small_gap_count = sum(1 for gap in gaps if gap <= 10.0)
    repeated_best = _recent_best_repetition_count(records) >= 2
    clear_or_close_top = small_gap_count >= min(2, len(gaps)) or any(
        gap > 10.0 for gap in gaps
    )
    return repeated_best and clear_or_close_top


def _broad_labels() -> set[str]:
    return {BROAD_STRATEGY_LABEL, LEGACY_P1_LABEL}


def _local_improve_labels() -> set[str]:
    return {LOCAL_IMPROVE_LABEL, LEGACY_P1C_LABEL}


def _strategy_labels() -> set[str]:
    return {
        BROAD_STRATEGY_LABEL,
        LOCAL_IMPROVE_LABEL,
        DUPLICATE_AUGMENT_LABEL,
        DUPLICATE_AUGMENT_REFINE_LABEL,
        RISK_TIER_DUPLICATE_LABEL,
        LOW_WILLINGNESS_DEEP_DUPLICATE_LABEL,
        TASK_RISK_DUPLICATE_LABEL,
        BUNDLE_SPLIT_DUPLICATE_LABEL,
        BUNDLE_MERGE_DUPLICATE_LABEL,
        PRESSURE_TARGETED_LABEL,
        PORTFOLIO_OVERLAY_LABEL,
        BEAM_STAGED_LABEL,
        LEGACY_P1_LABEL,
        LEGACY_P1C_LABEL,
    }


def _worst_case_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for result in _record_results(record):
            case_id = str(result.get("aggregate_metrics", {}).get("worst_case_id", ""))
            if case_id:
                counts[case_id] = counts.get(case_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _dominant_key(counts: dict[str, int]) -> str:
    return next(iter(counts), "")
