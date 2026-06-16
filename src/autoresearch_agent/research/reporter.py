"""Technical-report rendering helpers."""

from __future__ import annotations

from typing import Any


def render_report_stub() -> str:
    """Return a TODO report scaffold."""
    return "\n".join(
        [
            "# AutoResearch Report",
            "",
            "## Problem Understanding",
            "TODO",
            "",
            "## Strategy Search",
            "TODO",
            "",
            "## Experiments",
            "TODO",
            "",
            "## Final Solver",
            "TODO",
            "",
            "## Reflection",
            "TODO",
        ]
    )


def render_research_report(
    *,
    research_goal: str,
    case_path: str,
    solver_path: str,
    data_summary: dict[str, int | float],
    latest_experiment: dict[str, Any],
    baseline_suite: dict[str, Any],
    baseline_full_case: dict[str, Any],
    variant_results: list[dict[str, Any]],
    best_variant_path: str,
    best_variant_metrics: dict[str, Any],
    local_history_summary: dict[str, Any] | None = None,
    evidence_profile: dict[str, Any] | None = None,
    research_decision: dict[str, Any] | None = None,
    decision_source: str = "",
    llm_decision_raw: str = "",
    llm_decision_attempts: list[dict[str, Any]] | None = None,
    decision_validation_errors: list[str] | None = None,
    decision_evidence: list[str] | None = None,
    autopilot_prior: dict[str, Any] | None = None,
    solver_candidates: list[dict[str, Any]] | None = None,
    exported_solver_path: str = "",
    exported_solver_source_path: str = "",
    notes: list[str],
    llm_recommendations: list[str],
    llm_error: str = "",
) -> str:
    """Render a deterministic first-pass AutoResearch report."""
    local_history_summary = local_history_summary or {}
    evidence_profile = evidence_profile or {}
    research_decision = research_decision or {}
    autopilot_prior = autopilot_prior or {}
    llm_decision_attempts = llm_decision_attempts or []
    decision_validation_errors = decision_validation_errors or []
    decision_evidence = decision_evidence or []
    solver_candidates = solver_candidates or []
    affected_cases = [str(case) for case in evidence_profile.get("affected_cases", [])]
    evidence_reasons = [str(reason) for reason in evidence_profile.get("reasons", [])]
    run = latest_experiment.get("run", {})
    run_metrics = run.get("metrics", {})
    run_validation = run.get("validation", {})
    suite_aggregate = baseline_suite.get("aggregate_metrics", {})
    full_metrics = baseline_full_case.get("metrics", run_metrics)
    full_validation = baseline_full_case.get("validation", run_validation)
    next_experiments = llm_recommendations or [
        "Compare score-only greedy against probability-aware greedy.",
        "Try bundle-first and single-first ordering variants.",
        "Explore controlled duplicate dispatch for low-willingness tasks.",
        "Keep solver-side code Python 3.6 compatible and dependency-free.",
    ]
    lines = [
        "# AutoResearch Report",
        "",
        "## Goal",
        research_goal,
        "",
        "## Data Summary",
        f"- Case: `{case_path}`",
        f"- Candidate rows: `{data_summary.get('candidate_count', 0)}`",
        f"- Tasks: `{data_summary.get('task_count', 0)}`",
        f"- Couriers: `{data_summary.get('courier_count', 0)}`",
        f"- Bundled candidate ratio: `{data_summary.get('bundled_candidate_ratio', 0.0):.6f}`",
        "",
        "## Local History",
        f"- Local records: `{local_history_summary.get('record_count', 0)}`",
        f"- Variant-suite records: `{local_history_summary.get('variant_suite_record_count', 0)}`",
        f"- Broad strategy sweeps: `{local_history_summary.get('broad_strategy_sweep_count', 0)}`",
        f"- Local-improve sweeps: `{local_history_summary.get('local_improve_sweep_count', 0)}`",
        f"- Duplicate-augment sweeps: `{local_history_summary.get('duplicate_augment_sweep_count', 0)}`",
        f"- Risk-tier duplicate sweeps: `{local_history_summary.get('risk_tier_duplicate_sweep_count', 0)}`",
        f"- Task-risk duplicate sweeps: `{local_history_summary.get('task_risk_duplicate_sweep_count', 0)}`",
        f"- Bundle-merge duplicate sweeps: `{local_history_summary.get('bundle_merge_duplicate_sweep_count', 0)}`",
        f"- Latest local best: `{local_history_summary.get('latest_best_variant_path', '') or 'none'}`",
        f"- Local-improve diminishing returns: `{local_history_summary.get('local_improve_diminishing_returns', False)}`",
        f"- Duplicate-augment diminishing returns: `{local_history_summary.get('duplicate_augment_diminishing_returns', False)}`",
        f"- Risk-tier diminishing returns: `{local_history_summary.get('risk_tier_duplicate_diminishing_returns', False)}`",
        f"- Task-risk diminishing returns: `{local_history_summary.get('task_risk_duplicate_diminishing_returns', False)}`",
        f"- Bundle-merge diminishing returns: `{local_history_summary.get('bundle_merge_duplicate_diminishing_returns', False)}`",
        f"- Dominant worst case: `{local_history_summary.get('dominant_worst_case_id', '') or 'none'}`",
        "",
        "## Failure Mode Evidence",
        f"- Failure mode: `{evidence_profile.get('failure_mode', 'unknown')}`",
        f"- Strategy family: `{evidence_profile.get('strategy_family', 'unknown')}`",
        f"- Severity: `{float(evidence_profile.get('severity', 0.0)):.6f}`",
        f"- Evidence score: `{float(evidence_profile.get('evidence_score', 0.0)):.6f}`",
        f"- Affected cases: `{', '.join(affected_cases or ['none'])}`",
        "",
        "### Evidence Reasons",
        *[f"- {reason}" for reason in evidence_reasons],
        "",
        "## Evaluation Boundary",
        (
            "This agent loop reads local experiment history, generated solver "
            "outputs, and robust-suite proxy metrics only. External benchmark score records "
            "are not part of automatic decisions."
        ),
        "",
        "## Research Decision",
        f"- Decision source: `{decision_source or 'unknown'}`",
        f"- Search space: `{research_decision.get('search_space', 'unknown')}`",
        f"- Hypothesis: {research_decision.get('hypothesis', '')}",
        f"- Output dir: `{research_decision.get('output_dir', '') or 'none'}`",
        f"- Recommended solver: `{research_decision.get('recommended_solver_path', '') or 'none'}`",
        f"- Failure mode: `{research_decision.get('failure_mode', '') or 'unknown'}`",
        f"- Strategy family: `{research_decision.get('strategy_family', '') or 'unknown'}`",
        f"- Config IDs: `{research_decision.get('config_ids', [])}`",
        f"- Inline config count: `{len(research_decision.get('configs', []))}`",
        "",
        "## Decision Guardrail",
        "- Invalid LLM decisions are rejected after one structured repair attempt.",
        "- No hand-coded decision is substituted when LLM decision validation fails.",
        "",
        "## Local Autopilot Plan",
        f"- Search space: `{autopilot_prior.get('search_space', 'unknown')}`",
        f"- Failure mode: `{autopilot_prior.get('failure_mode', '') or 'unknown'}`",
        f"- Config IDs: `{autopilot_prior.get('config_ids', [])}`",
        "",
        "### Decision Evidence",
        *[f"- {item}" for item in decision_evidence],
        "",
        "### LLM Decision Diagnostics",
        f"- Raw decision: `{_compact_text(llm_decision_raw) or 'none'}`",
        *[
            (
                f"- Attempt {attempt.get('attempt', 0)} valid="
                f"`{attempt.get('is_valid', False)}` errors="
                f"`{attempt.get('validation_errors', [])}`"
            )
            for attempt in llm_decision_attempts
        ],
        *[
            f"- Validation error: {error}"
            for error in (decision_validation_errors or ["none"])
        ],
        "",
        "## Baseline Suite Experiment",
        f"- Solver: `{solver_path}`",
        f"- Suite valid: `{suite_aggregate.get('is_valid', False)}`",
        f"- Cases: `{suite_aggregate.get('case_count', 0)}`",
        f"- Invalid cases: `{suite_aggregate.get('invalid_case_count', 0)}`",
        f"- Timeout cases: `{suite_aggregate.get('timeout_count', 0)}`",
        f"- Mean coverage: `{suite_aggregate.get('mean_task_coverage_ratio', 0.0):.6f}`",
        f"- Mean expected success: `{suite_aggregate.get('mean_expected_success_ratio', 0.0):.6f}`",
        f"- Mean proxy score: `{suite_aggregate.get('mean_proxy_score', 0.0):.6f}`",
        f"- Worst case: `{suite_aggregate.get('worst_case_id', '')}`",
        "",
        "## Full Case Reference",
        f"- Valid: `{full_validation.get('is_valid', False)}`",
        f"- Assigned tasks: `{full_metrics.get('assigned_task_count', 0)}`",
        f"- Unassigned tasks: `{full_metrics.get('unassigned_task_count', 0)}`",
        f"- Total score: `{full_metrics.get('total_score', 0.0):.6f}`",
        f"- Expected success ratio: `{full_metrics.get('expected_success_ratio', 0.0):.6f}`",
        f"- Proxy score: `{full_metrics.get('proxy_score', 0.0):.6f}`",
        "",
        "## Variant Suite Leaderboard",
        f"- Best variant: `{best_variant_path or 'none'}`",
        f"- Best mean proxy score: `{best_variant_metrics.get('mean_proxy_score', best_variant_metrics.get('proxy_score', 0.0)):.6f}`",
        "",
        "| Rank | Variant | Valid | Cases | Timeouts | Mean Coverage | Mean Expected Success | Mean Proxy | Worst Case |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_render_variant_rows(variant_results))
    lines.extend(
        [
            "",
            "## Recommended Candidates",
            f"- Exported solver: `{exported_solver_path or 'none'}`",
            f"- Export source: `{exported_solver_source_path or 'none'}`",
            "",
            "| Reason | Rank | Family | Pipeline | Mean Proxy | Output Sig | Variant |",
            "| --- | ---: | --- | --- | ---: | --- | --- |",
            *_render_candidate_rows(solver_candidates),
        ]
    )
    lines.extend(
        [
            "",
            "## Current Notes",
            *[f"- {note}" for note in notes],
            "",
            "## LLM Recommendations",
            *[f"- {recommendation}" for recommendation in next_experiments],
        ]
    )
    if llm_error:
        lines.extend(["", f"LLM recommendation error: `{llm_error}`"])
    lines.extend(
        [
            "",
            "## Next Experiments",
            *[f"- {recommendation}" for recommendation in next_experiments],
            "",
        ]
    )
    return "\n".join(lines)


def _render_variant_rows(variant_results: list[dict[str, Any]]) -> list[str]:
    """Render variant result rows for the report leaderboard."""
    if not variant_results:
        return [
            "| 0 | `none` | False | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | `none` |"
        ]
    rows = []
    for result in variant_results:
        aggregate = result.get("aggregate_metrics")
        if isinstance(aggregate, dict):
            rows.append(
                "| "
                f"{result.get('rank', 0)} | "
                f"`{result.get('variant_path', '')}` | "
                f"{aggregate.get('is_valid', False)} | "
                f"{aggregate.get('case_count', 0)} | "
                f"{aggregate.get('timeout_count', 0)} | "
                f"{aggregate.get('mean_task_coverage_ratio', 0.0):.6f} | "
                f"{aggregate.get('mean_expected_success_ratio', 0.0):.6f} | "
                f"{aggregate.get('mean_proxy_score', 0.0):.6f} | "
                f"`{aggregate.get('worst_case_id', '')}` |"
            )
        else:
            metrics = result.get("metrics", {})
            rows.append(
                "| "
                f"{result.get('rank', 0)} | "
                f"`{result.get('variant_path', '')}` | "
                f"{metrics.get('is_valid', False)} | "
                "1 | "
                "0 | "
                f"{metrics.get('task_coverage_ratio', 0.0):.6f} | "
                f"{metrics.get('expected_success_ratio', 0.0):.6f} | "
                f"{metrics.get('proxy_score', 0.0):.6f} | "
                "`single-run` |"
            )
    return rows


def _render_candidate_rows(candidates: list[dict[str, Any]]) -> list[str]:
    """Render recommended solver candidates for the report."""
    if not candidates:
        return ["| `none` | 0 | `none` | `none` | 0.000000 | `none` | `none` |"]
    rows = []
    for candidate in candidates:
        rows.append(
            "| "
            f"`{candidate.get('reason', '')}` | "
            f"{candidate.get('rank', 0)} | "
            f"`{candidate.get('family', '')}` | "
            f"`{candidate.get('pipeline', '')}` | "
            f"{float(candidate.get('mean_proxy_score', 0.0)):.6f} | "
            f"`{candidate.get('output_signature', '') or 'none'}` | "
            f"`{candidate.get('variant_path', '')}` |"
        )
    return rows


def _compact_text(value: str, *, limit: int = 500) -> str:
    """Return one-line diagnostic text suitable for report display."""
    compacted = " ".join(value.strip().split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3] + "..."
