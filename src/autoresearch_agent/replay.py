"""Replay a saved AutoResearch trajectory without calling an LLM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPLAY_SCHEMA_VERSION = 1


def build_replay_fixture(
    *,
    final_state: Mapping[str, Any],
    run_id: str = "",
    iteration: int = 1,
    source: str = "local_llm_run",
) -> dict[str, Any]:
    """Build a compact replay fixture from one completed graph state."""
    latest_experiment = _as_mapping(final_state.get("latest_experiment", {}))
    latest_run_id = run_id or str(latest_experiment.get("run_id", ""))
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source": source,
        "description": (
            "Sanitized replay fixture captured from one LLM-guided "
            "AutoResearch iteration. Replaying this file does not call an LLM "
            "or execute solver experiments."
        ),
        "iterations": [
            {
                "iteration": iteration,
                "run_id": latest_run_id,
                "research_goal": final_state.get("research_goal", ""),
                "case_path": final_state.get("case_path", ""),
                "solver_path": final_state.get("solver_path", ""),
                "data_summary": _as_mapping(final_state.get("data_summary", {})),
                "local_history_summary": _compact_history(
                    _as_mapping(final_state.get("local_history_summary", {}))
                ),
                "evidence_profile": _as_mapping(
                    final_state.get("evidence_profile", {})
                ),
                "decision_source": final_state.get("decision_source", ""),
                "research_decision": _as_mapping(
                    final_state.get("research_decision", {})
                ),
                "decision_evidence": _string_list(
                    final_state.get("decision_evidence", [])
                ),
                "llm_decision_attempts": _compact_attempts(
                    final_state.get("llm_decision_attempts", [])
                ),
                "decision_validation_errors": _string_list(
                    final_state.get("decision_validation_errors", [])
                ),
                "baseline_aggregate_metrics": _as_mapping(
                    _as_mapping(final_state.get("baseline_suite", {})).get(
                        "aggregate_metrics",
                        {},
                    )
                ),
                "best_variant_path": final_state.get("best_variant_path", ""),
                "best_variant_metrics": _as_mapping(
                    final_state.get("best_variant_metrics", {})
                ),
                "solver_candidates": _compact_candidates(
                    final_state.get("solver_candidates", [])
                ),
                "exported_solver_path": final_state.get("exported_solver_path", ""),
                "exported_solver_source_path": final_state.get(
                    "exported_solver_source_path",
                    "",
                ),
                "llm_recommendations": _string_list(
                    final_state.get("llm_recommendations", [])
                ),
                "notes": _string_list(final_state.get("notes", []))[:8],
            }
        ],
    }


def load_replay_fixture(path: str | Path) -> dict[str, Any]:
    """Load and validate a replay fixture."""
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("replay fixture must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != REPLAY_SCHEMA_VERSION:
        raise ValueError(
            "unsupported replay schema version: "
            f"{schema_version!r}; expected {REPLAY_SCHEMA_VERSION}"
        )
    iterations = payload.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("replay fixture requires a non-empty iterations list")
    return payload


def render_replay_markdown(fixture: Mapping[str, Any]) -> str:
    """Render a replay fixture as a no-key Markdown trajectory."""
    iterations = [
        _as_mapping(item) for item in _as_sequence(fixture.get("iterations", []))
    ]
    lines = [
        "# AutoResearch Replay",
        "",
        f"- Source: `{fixture.get('source', 'unknown')}`",
        f"- Schema version: `{fixture.get('schema_version', 'unknown')}`",
        f"- Iterations: `{len(iterations)}`",
        "",
        str(fixture.get("description", "")).strip(),
        "",
    ]
    for item in iterations:
        lines.extend(_render_iteration(item))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Run the replay command-line interface."""
    args = _parse_args()
    fixture = load_replay_fixture(args.fixture)
    if args.json:
        print(json.dumps(fixture, ensure_ascii=False, indent=2))
        return
    print(render_replay_markdown(fixture), end="")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved AutoResearch trajectory without calling an LLM or "
            "running solver experiments."
        )
    )
    parser.add_argument("fixture", help="Path to a replay fixture JSON file.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the normalized fixture JSON instead of Markdown.",
    )
    return parser.parse_args()


def _render_iteration(item: Mapping[str, Any]) -> list[str]:
    data_summary = _as_mapping(item.get("data_summary", {}))
    history = _as_mapping(item.get("local_history_summary", {}))
    evidence = _as_mapping(item.get("evidence_profile", {}))
    decision = _as_mapping(item.get("research_decision", {}))
    baseline = _as_mapping(item.get("baseline_aggregate_metrics", {}))
    best_metrics = _as_mapping(item.get("best_variant_metrics", {}))
    candidates = [
        _as_mapping(candidate)
        for candidate in _as_sequence(item.get("solver_candidates", []))
    ]
    attempts = [
        _as_mapping(attempt)
        for attempt in _as_sequence(item.get("llm_decision_attempts", []))
    ]
    lines = [
        f"## Iteration {item.get('iteration', 0)}",
        "",
        f"- Run ID: `{item.get('run_id', '') or 'none'}`",
        f"- Goal: {item.get('research_goal', '')}",
        f"- Case: `{item.get('case_path', '')}`",
        f"- Solver baseline: `{item.get('solver_path', '')}`",
        "",
        "### Data",
        f"- Candidates: `{data_summary.get('candidate_count', 0)}`",
        f"- Tasks: `{data_summary.get('task_count', 0)}`",
        f"- Couriers: `{data_summary.get('courier_count', 0)}`",
        f"- Bundled candidate ratio: `{_float(data_summary.get('bundled_candidate_ratio', 0.0)):.6f}`",
        "",
        "### Local Evidence",
        f"- Prior records: `{history.get('record_count', 0)}`",
        f"- Failure mode: `{evidence.get('failure_mode', 'unknown')}`",
        f"- Strategy family: `{evidence.get('strategy_family', 'unknown')}`",
        f"- Evidence score: `{_float(evidence.get('evidence_score', 0.0)):.6f}`",
        "",
        "### Decision",
        f"- Source: `{item.get('decision_source', '') or 'unknown'}`",
        f"- Search space: `{decision.get('search_space', '') or 'unknown'}`",
        f"- Hypothesis: {decision.get('hypothesis', '')}",
        f"- Config IDs: `{_preview_values(decision.get('config_ids', []))}`",
        f"- Candidate limit: `{decision.get('candidate_limit', 0)}`",
        f"- Failure mode: `{decision.get('failure_mode', '') or 'unknown'}`",
        "",
        "### Guardrail",
        *[
            (
                f"- Attempt {attempt.get('attempt', 0)} valid="
                f"`{attempt.get('is_valid', False)}` errors="
                f"`{attempt.get('validation_errors', [])}`"
            )
            for attempt in attempts
        ],
        *[
            f"- Validation error: {error}"
            for error in _string_list(item.get("decision_validation_errors", []))
        ],
        "",
        "### Evaluation",
        f"- Baseline valid: `{baseline.get('is_valid', False)}`",
        f"- Baseline cases: `{baseline.get('case_count', 0)}`",
        f"- Baseline mean proxy: `{_float(baseline.get('mean_proxy_score', 0.0)):.6f}`",
        f"- Best variant: `{item.get('best_variant_path', '') or 'none'}`",
        f"- Best mean proxy: `{_float(best_metrics.get('mean_proxy_score', best_metrics.get('proxy_score', 0.0))):.6f}`",
        f"- Exported solver: `{item.get('exported_solver_path', '') or 'none'}`",
        "",
        "### Candidate Preview",
    ]
    if candidates:
        lines.extend(
            [
                (
                    f"- Rank `{candidate.get('rank', 0)}` "
                    f"{candidate.get('family', 'unknown')} / "
                    f"{candidate.get('pipeline', candidate.get('intent', 'unknown'))}: "
                    f"`{candidate.get('variant_path', '')}`"
                )
                for candidate in candidates[:5]
            ]
        )
    else:
        lines.append("- none")
    recommendations = _string_list(item.get("llm_recommendations", []))
    if recommendations:
        lines.extend(["", "### Recommendations"])
        lines.extend(f"- {recommendation}" for recommendation in recommendations[:5])
    notes = _string_list(item.get("notes", []))
    if notes:
        lines.extend(["", "### Notes"])
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return lines


def _compact_history(history: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "record_count",
        "variant_suite_record_count",
        "broad_strategy_sweep_count",
        "local_improve_sweep_count",
        "duplicate_augment_sweep_count",
        "risk_tier_duplicate_sweep_count",
        "task_risk_duplicate_sweep_count",
        "bundle_merge_duplicate_sweep_count",
        "dominant_worst_case_id",
        "evidence_profile",
    ]
    return {key: history.get(key) for key in keys if key in history}


def _compact_attempts(value: Any) -> list[dict[str, Any]]:
    attempts = []
    for item in _as_sequence(value):
        attempt = _as_mapping(item)
        attempts.append(
            {
                "attempt": attempt.get("attempt", 0),
                "is_valid": attempt.get("is_valid", False),
                "validation_errors": _string_list(
                    attempt.get("validation_errors", [])
                ),
            }
        )
    return attempts


def _compact_candidates(value: Any) -> list[dict[str, Any]]:
    candidates = []
    for item in _as_sequence(value):
        candidate = _as_mapping(item)
        candidates.append(
            {
                "reason": candidate.get("reason", ""),
                "rank": candidate.get("rank", 0),
                "variant_path": candidate.get("variant_path", ""),
                "family": candidate.get("family", ""),
                "intent": candidate.get("intent", ""),
                "pipeline": candidate.get("pipeline", ""),
                "mean_proxy_score": candidate.get("mean_proxy_score", 0.0),
                "output_signature": candidate.get("output_signature", ""),
            }
        )
    return candidates


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_sequence(value) if str(item).strip()]


def _preview_values(value: Any, *, limit: int = 5) -> list[str]:
    items = _string_list(value)
    if len(items) <= limit:
        return items
    return [*items[:limit], f"... +{len(items) - limit} more"]


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


if __name__ == "__main__":
    main()
