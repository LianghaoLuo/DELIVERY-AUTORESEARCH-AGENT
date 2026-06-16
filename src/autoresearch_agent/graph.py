"""LangGraph entrypoint for the offline AutoResearch agent."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from autoresearch_agent.context import Context
from autoresearch_agent.prompts import (
    LLM_DECISION_PROMPT,
    LLM_DECISION_REPAIR_PROMPT,
    RESEARCH_RECOMMENDATION_PROMPT,
)
from autoresearch_agent.research.autopilot import (
    ResearchDecision,
    decide_next_experiment,
    research_decision_to_dict,
)
from autoresearch_agent.research.llm_decision import (
    llm_decision_allowed_values,
    validate_llm_research_decision,
)
from autoresearch_agent.research.planner import propose_strategy_plan
from autoresearch_agent.research.strategy_space import (
    strategy_config_ids_for_region,
    strategy_sweep_label,
)
from autoresearch_agent.solver_dev.case_suite import CaseSuiteName
from autoresearch_agent.state import InputState, State
from autoresearch_agent.tools import (
    export_solver_candidate,
    get_strategy_catalog,
    get_strategy_primitive_schema,
    load_experiment_history,
    run_stop_search_evaluation,
    run_strategy_evaluation,
    summarize_candidate_data,
    write_research_report,
)
from autoresearch_agent.utils import get_message_text, load_chat_model

AGENT_VERSION = "strategy_loop_v3"


async def plan_research(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Read local inputs and prepare the baseline research plan."""
    goal = state.research_goal or runtime.context.research_goal
    case_path = _select_case_path(state, runtime.context)
    solver_path = runtime.context.solver_entrypoint
    case_summary = summarize_candidate_data(data_file=case_path)
    data_summary = case_summary["data_summary"]
    strategy_plan = propose_strategy_plan()
    return {
        "research_goal": goal,
        "case_path": case_summary["case_path"],
        "solver_path": solver_path,
        "data_summary": data_summary,
        "notes": [
            *state.notes,
            f"Loaded case {case_path} with {data_summary['candidate_count']} candidates.",
            f"Detected {data_summary['task_count']} tasks and {data_summary['courier_count']} couriers.",
            f"Initial strategy plan: {'; '.join(strategy_plan)}",
        ],
    }


async def load_local_history(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Load local experiment history for deterministic strategy planning."""
    history = load_experiment_history(experiments_dir=runtime.context.experiments_dir)
    records = history["experiment_records"]
    history_summary = history["local_history_summary"]
    evidence_profile = history["evidence_profile"]
    return {
        "experiment_records": records,
        "local_history_summary": history_summary,
        "evidence_profile": evidence_profile,
        "notes": [
            *state.notes,
            (
                "Loaded local experiment history: "
                f"records={history_summary['record_count']}, "
                f"broad_sweeps={history_summary['broad_strategy_sweep_count']}, "
                f"local_improve_sweeps={history_summary['local_improve_sweep_count']}, "
                f"duplicate_augment_sweeps={history_summary['duplicate_augment_sweep_count']}, "
                f"risk_tier_sweeps={history_summary['risk_tier_duplicate_sweep_count']}, "
                f"task_risk_sweeps={history_summary['task_risk_duplicate_sweep_count']}, "
                f"bundle_merge_sweeps={history_summary['bundle_merge_duplicate_sweep_count']}."
            ),
        ],
    }


async def decide_next_experiment_node(
    state: State, runtime: Runtime[Context]
) -> dict[str, Any]:
    """Choose the next local research action via a guarded decision path."""
    catalog = get_strategy_catalog()
    autopilot_prior: dict[str, Any] = {}
    decision: ResearchDecision
    decision_source: str
    llm_decision_raw = ""
    llm_decision_attempts: list[dict[str, Any]] = []
    validation_errors: list[str] = []

    if runtime.context.enable_llm_decisions:
        llm_result = await _try_llm_research_decision(
            state,
            runtime,
            strategy_catalog=catalog,
        )
        llm_decision_raw = llm_result["raw_text"]
        llm_decision_attempts = list(llm_result["attempts"])
        validation_errors = list(llm_result["errors"])
        maybe_decision = llm_result["decision"]
        if not isinstance(maybe_decision, ResearchDecision):
            raise ValueError(
                "LLM research decision was rejected by the decision guardrail: "
                + "; ".join(validation_errors or ["no valid decision produced"])
            )
        decision = maybe_decision
        decision_source = str(llm_result["accepted_source"])
    else:
        autopilot_decision = _decision_with_catalog_selection(
            decide_next_experiment(state.local_history_summary)
        )
        autopilot_prior = research_decision_to_dict(autopilot_decision)
        decision = autopilot_decision
        decision_source = "local_autopilot"
        validation_errors = []

    decision_payload = research_decision_to_dict(decision)
    evidence = [
        f"decision_source={decision_source}",
        f"search_space={decision.search_space}",
        f"hypothesis={decision.hypothesis}",
        f"failure_mode={decision.failure_mode}",
        f"strategy_family={decision.strategy_family}",
        f"config_ids={decision.selected_config_ids or []}",
        f"inline_config_count={len(decision.configs or [])}",
        *[f"validation_error={error}" for error in validation_errors],
        *decision.reasons,
    ]
    return {
        "research_decision": decision_payload,
        "decision_source": decision_source,
        "llm_decision_raw": llm_decision_raw,
        "llm_decision_attempts": llm_decision_attempts,
        "decision_validation_errors": validation_errors,
        "autopilot_prior": autopilot_prior,
        "strategy_catalog": catalog,
        "decision_evidence": evidence,
        "notes": [
            *state.notes,
            (
                "Research decision: "
                f"source={decision_source}; search_space={decision.search_space}; "
                f"hypothesis: {decision.hypothesis}"
            ),
        ],
    }


async def _try_llm_research_decision(
    state: State,
    runtime: Runtime[Context],
    *,
    strategy_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the configured LLM for a structured decision and validate it."""
    attempts: list[dict[str, Any]] = []
    try:
        model = load_chat_model(
            runtime.context.model,
            base_url=runtime.context.model_base_url,
            api_key_env=runtime.context.model_api_key_env,
        )
        prompt = LLM_DECISION_PROMPT.format(
            research_goal=state.research_goal or runtime.context.research_goal,
            data_summary=json.dumps(
                state.data_summary, ensure_ascii=False, sort_keys=True
            ),
            local_history_summary=json.dumps(
                state.local_history_summary,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            evidence_profile=json.dumps(
                state.evidence_profile,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            strategy_catalog=json.dumps(
                strategy_catalog,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            primitive_schema=json.dumps(
                get_strategy_primitive_schema(),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )
        response = await model.ainvoke(
            [
                {"role": "system", "content": runtime.context.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        raw_text = get_message_text(response)
        validation = validate_llm_research_decision(raw_text)
        attempts.append(_llm_decision_attempt(1, validation))
        if validation.is_valid:
            return {
                "decision": validation.decision,
                "raw_text": validation.raw_text,
                "errors": [],
                "attempts": attempts,
                "accepted_source": "llm",
            }
        repair_prompt = LLM_DECISION_REPAIR_PROMPT.format(
            validation_errors=json.dumps(
                validation.errors,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            raw_decision=validation.raw_text,
            allowed_values=json.dumps(
                llm_decision_allowed_values(
                    str(validation.payload.get("search_space", ""))
                    if isinstance(validation.payload, dict)
                    else ""
                ),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )
        repair_response = await model.ainvoke(
            [
                {"role": "system", "content": runtime.context.system_prompt},
                {"role": "user", "content": repair_prompt},
            ]
        )
        repair_raw_text = get_message_text(repair_response)
        repair_validation = validate_llm_research_decision(repair_raw_text)
        attempts.append(_llm_decision_attempt(2, repair_validation))
        if repair_validation.is_valid:
            return {
                "decision": repair_validation.decision,
                "raw_text": repair_validation.raw_text,
                "errors": [],
                "attempts": attempts,
                "accepted_source": "llm_repair",
            }
        return {
            "decision": None,
            "raw_text": repair_validation.raw_text,
            "errors": [*validation.errors, *repair_validation.errors],
            "attempts": attempts,
            "accepted_source": "",
        }
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return {
            "decision": None,
            "raw_text": "",
            "errors": [f"LLM decision failed: {type(exc).__name__}: {exc}"],
            "attempts": attempts,
            "accepted_source": "",
        }


def _llm_decision_attempt(
    attempt: int,
    validation: Any,
) -> dict[str, Any]:
    """Return compact diagnostics for one LLM decision attempt."""
    return {
        "attempt": attempt,
        "raw_text": validation.raw_text,
        "validation_errors": list(validation.errors),
        "is_valid": validation.is_valid,
    }


async def run_selected_experiment(
    state: State, runtime: Runtime[Context]
) -> dict[str, Any]:
    """Run the experiment selected by the local AutoResearch controller."""
    case_path = state.case_path or str(_select_case_path(state, runtime.context))
    solver_path = state.solver_path or runtime.context.solver_entrypoint
    timeout_seconds = float(runtime.context.case_timeout_seconds)
    decision = state.research_decision or {}
    search_space = str(decision.get("search_space", "broad_strategy"))
    selected_config_ids = _string_list(decision.get("config_ids", []))
    inline_configs = decision.get("configs", [])
    if not isinstance(inline_configs, list):
        inline_configs = []
    if search_space == "stop_search":
        recommended_solver = str(decision.get("recommended_solver_path", ""))
        if recommended_solver:
            evaluation = run_stop_search_evaluation(
                solver_path=recommended_solver,
                data_file=case_path,
                baseline_solver=solver_path,
                experiments_dir=runtime.context.experiments_dir,
                timeout_seconds=timeout_seconds,
                provenance=_agent_provenance(state, runtime, action="stop_evidence"),
                evidence_profile=state.evidence_profile,
                case_suite=_case_suite_name(runtime.context.local_case_suite),
            )
            selected_label = "graph-auto-stop-evidence"
        else:
            evaluation = run_strategy_evaluation(
                search_space="bundle_merge_duplicate",
                data_file=case_path,
                baseline_solver=solver_path,
                experiments_dir=runtime.context.experiments_dir,
                timeout_seconds=timeout_seconds,
                candidate_limit=int(decision.get("candidate_limit", 4)),
                provenance=_agent_provenance(
                    state,
                    runtime,
                    action="bundle_merge_duplicate_sweep",
                ),
                hypothesis=str(decision.get("hypothesis", "")),
                selected_config_ids=selected_config_ids,
                inline_configs=inline_configs,
                evidence_profile=state.evidence_profile,
                case_suite=_case_suite_name(runtime.context.local_case_suite),
            )
            selected_label = strategy_sweep_label("bundle_merge_duplicate")
    else:
        evaluation = run_strategy_evaluation(
            search_space=search_space,
            data_file=case_path,
            baseline_solver=solver_path,
            experiments_dir=runtime.context.experiments_dir,
            output_dir=str(decision.get("output_dir", "")) or None,
            timeout_seconds=timeout_seconds,
            candidate_limit=int(decision.get("candidate_limit", 4)),
            provenance=_agent_provenance(
                state,
                runtime,
                action=f"{search_space}_sweep",
            ),
            hypothesis=str(decision.get("hypothesis", "")),
            selected_config_ids=selected_config_ids,
            inline_configs=inline_configs,
            evidence_profile=state.evidence_profile,
            case_suite=_case_suite_name(runtime.context.local_case_suite),
        )
        selected_label = strategy_sweep_label(search_space)

    history = load_experiment_history(experiments_dir=runtime.context.experiments_dir)
    baseline_aggregate = evaluation.baseline_suite["aggregate_metrics"]
    batch_payload = evaluation.batch_payload
    variant_results = batch_payload["variant_results"]
    best_variant_path = evaluation.best_variant_path
    return {
        "latest_experiment": evaluation.latest_record,
        "experiment_records": history["experiment_records"],
        "baseline_suite": evaluation.baseline_suite,
        "baseline_full_case": evaluation.baseline_full_case,
        "variant_results": variant_results,
        "auto_sweep_results": evaluation.leaderboard_rows,
        "solver_candidates": evaluation.solver_candidates,
        "best_variant_path": best_variant_path,
        "best_variant_metrics": evaluation.best_variant_metrics,
        "best_solver_path": best_variant_path
        or (solver_path if baseline_aggregate["is_valid"] else ""),
        "experiment_log": [
            *state.experiment_log,
            (
                "Ran baseline solver suite: "
                f"valid={baseline_aggregate['is_valid']}, "
                f"cases={baseline_aggregate['case_count']}, "
                f"mean_proxy_score={baseline_aggregate['mean_proxy_score']:.6f}."
            ),
            (
                "Ran selected AutoResearch action: "
                f"label={selected_label}, "
                f"search_space={search_space}, "
                f"best={best_variant_path or 'none'}, "
                f"result_count={len(variant_results)}."
            ),
        ],
    }


async def export_suggested_solver(
    state: State, runtime: Runtime[Context]
) -> dict[str, Any]:
    """Export the current best local candidate without replacing solver.py."""
    source_path = state.best_variant_path or state.best_solver_path
    if not source_path:
        return {
            "exported_solver_path": "",
            "exported_solver_source_path": "",
            "notes": [*state.notes, "No suggested solver was exported."],
        }
    export_result = export_solver_candidate(
        source_path=source_path,
        destination_path=runtime.context.suggested_solver_output_path,
    )
    return {
        "exported_solver_path": export_result["exported_solver_path"],
        "exported_solver_source_path": export_result["exported_solver_source_path"],
        "notes": [
            *state.notes,
            (
                "Exported suggested solver candidate: "
                f"{source_path} -> {export_result['exported_solver_path']}."
            ),
        ],
    }


async def generate_research_suggestions(
    state: State, runtime: Runtime[Context]
) -> dict[str, Any]:
    """Ask the configured LLM for the next local solver experiments."""
    if not runtime.context.enable_llm_recommendations:
        fallback = [
            "LLM disabled: compare score-only greedy against probability-aware greedy.",
            "LLM disabled: test bundle-first and single-first ordering variants.",
            "LLM disabled: evaluate controlled duplicate dispatch for low-willingness tasks.",
        ]
        return {
            "llm_recommendations": fallback,
            "llm_recommendation_text": "\n".join(f"- {item}" for item in fallback),
            "llm_error": "",
            "notes": [*state.notes, "LLM recommendations were disabled for this run."],
        }

    try:
        model = load_chat_model(
            runtime.context.model,
            base_url=runtime.context.model_base_url,
            api_key_env=runtime.context.model_api_key_env,
        )
        prompt = RESEARCH_RECOMMENDATION_PROMPT.format(
            research_goal=state.research_goal or runtime.context.research_goal,
            data_summary=json.dumps(
                state.data_summary, ensure_ascii=False, sort_keys=True
            ),
            case_path=state.case_path,
            latest_experiment=json.dumps(
                state.latest_experiment,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            variant_leaderboard=json.dumps(
                state.variant_results,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            experiment_count=len(state.experiment_records),
        )
        response = await model.ainvoke(
            [
                {"role": "system", "content": runtime.context.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        text = get_message_text(response)
        recommendations = _parse_recommendations(text)
        return {
            "llm_recommendations": recommendations,
            "llm_recommendation_text": text,
            "llm_error": "",
            "notes": [
                *state.notes,
                f"LLM proposed {len(recommendations)} next experiment(s).",
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive recommendation backup
        error = f"{type(exc).__name__}: {exc}"
        fallback = [
            "LLM unavailable: compare score-only greedy against probability-aware greedy.",
            "LLM unavailable: test bundle-first and single-first ordering variants.",
            "LLM unavailable: evaluate controlled duplicate dispatch for low-willingness tasks.",
        ]
        return {
            "llm_recommendations": fallback,
            "llm_recommendation_text": "\n".join(f"- {item}" for item in fallback),
            "llm_error": error,
            "notes": [*state.notes, f"LLM recommendation step failed: {error}"],
        }


async def write_report(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Write a deterministic report from the latest local experiment."""
    return write_research_report(
        context=runtime.context,
        report_path=runtime.context.report_path,
        research_goal=state.research_goal or runtime.context.research_goal,
        case_path=state.case_path,
        solver_path=state.solver_path or runtime.context.solver_entrypoint,
        data_summary=state.data_summary,
        latest_experiment=state.latest_experiment,
        baseline_suite=state.baseline_suite,
        baseline_full_case=state.baseline_full_case,
        variant_results=state.variant_results,
        best_variant_path=state.best_variant_path,
        best_variant_metrics=state.best_variant_metrics,
        local_history_summary=state.local_history_summary,
        evidence_profile=state.evidence_profile,
        research_decision=state.research_decision,
        decision_source=state.decision_source,
        llm_decision_raw=state.llm_decision_raw,
        llm_decision_attempts=state.llm_decision_attempts,
        decision_validation_errors=state.decision_validation_errors,
        decision_evidence=state.decision_evidence,
        autopilot_prior=state.autopilot_prior,
        solver_candidates=state.solver_candidates,
        exported_solver_path=state.exported_solver_path,
        exported_solver_source_path=state.exported_solver_source_path,
        notes=[*state.notes, *state.experiment_log],
        llm_recommendations=state.llm_recommendations,
        llm_error=state.llm_error,
    )


def _select_case_path(state: State, context: Context) -> Path:
    """Choose the local case to use for the current research run."""
    if state.data_files:
        return Path(state.data_files[0])
    return Path(context.data_dir) / "large_seed301.txt"


def _case_suite_name(value: str) -> CaseSuiteName:
    """Return a supported local case-suite name from runtime context."""
    _ = value
    return "robust"


def _parse_recommendations(text: str) -> list[str]:
    """Extract concise recommendation lines from an LLM response."""
    recommendations: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("-*0123456789. )\t").strip()
        line = line.replace("**", "").strip()
        if line:
            recommendations.append(line)
        if len(recommendations) >= 5:
            break
    if recommendations:
        return recommendations
    stripped = text.strip()
    return [stripped] if stripped else []


def _decision_with_catalog_selection(decision: ResearchDecision) -> ResearchDecision:
    """Attach concrete catalog selectors to the local autopilot plan when possible."""
    if (
        decision.search_space == "stop_search"
        or decision.selected_config_ids
        or decision.configs
    ):
        return decision
    try:
        config_ids = strategy_config_ids_for_region(
            decision.search_space,
            decision.param_region,
            fallback_to_all=True,
        )
    except ValueError:
        return decision
    return replace(
        decision,
        selected_config_ids=config_ids,
    )


def _string_list(value: Any) -> list[str]:
    """Return non-empty string items from a JSON-ish list field."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _agent_provenance(
    state: State,
    runtime: Runtime[Context],
    *,
    action: str,
) -> dict[str, Any]:
    """Return provenance metadata for records created by the graph loop."""
    return {
        "origin": "agent_loop",
        "agent_version": AGENT_VERSION,
        "selected_by": state.decision_source or "unknown",
        "accepted_source": state.decision_source or "unknown",
        "action": action,
        "research_goal": state.research_goal or runtime.context.research_goal,
        "history_scope": runtime.context.experiments_dir,
        "decision": state.research_decision,
        "autopilot_prior": state.autopilot_prior,
        "llm_decision_raw": state.llm_decision_raw,
        "validation_errors": list(state.decision_validation_errors),
        "llm_decision_attempts": list(state.llm_decision_attempts),
    }


builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node(plan_research)
builder.add_node(load_local_history)
builder.add_node(decide_next_experiment_node)
builder.add_node(run_selected_experiment)
builder.add_node(export_suggested_solver)
builder.add_node(generate_research_suggestions)
builder.add_node(write_report)

builder.add_edge("__start__", "plan_research")
builder.add_edge("plan_research", "load_local_history")
builder.add_edge("load_local_history", "decide_next_experiment_node")
builder.add_edge("decide_next_experiment_node", "run_selected_experiment")
builder.add_edge("run_selected_experiment", "export_suggested_solver")
builder.add_edge("export_suggested_solver", "generate_research_suggestions")
builder.add_edge("generate_research_suggestions", "write_report")
builder.add_edge("write_report", "__end__")

graph = builder.compile(name="Delivery AutoResearch Agent")
