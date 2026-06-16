"""Typed local tools for the AutoResearch agent.

Tools in this module may use LangGraph/LangChain-facing research helpers and
local experiment runners. They must never be imported by ``solvers/solver.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from autoresearch_agent.context import Context
from autoresearch_agent.research.autopilot import summarize_local_history
from autoresearch_agent.research.evaluator import (
    LocalEvaluationResult,
    evaluate_stop_search_candidate,
    evaluate_strategy_space,
)
from autoresearch_agent.research.experiment_store import ExperimentStore
from autoresearch_agent.research.reporter import render_research_report
from autoresearch_agent.research.strategy_space import (
    strategy_primitive_schema,
    strategy_space_catalog,
)
from autoresearch_agent.solver_dev.case_suite import CaseSuiteName
from autoresearch_agent.solver_dev.packager import package_solver
from autoresearch_agent.solver_dev.parser import (
    parse_candidate_file,
    summarize_candidate_table,
)

JsonSchema = dict[str, Any]
ToolCallable = Callable[..., Any]


@dataclass(frozen=True)
class AgentTool:
    """One local tool exposed through the agent tool registry."""

    name: str
    description: str
    args_schema: JsonSchema
    invoke: ToolCallable

    def as_openai_tool(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool descriptor."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema,
            },
        }

    def __call__(self, **kwargs: Any) -> Any:
        """Invoke the wrapped local tool."""
        return self.invoke(**kwargs)


def summarize_candidate_data(*, data_file: str | Path) -> dict[str, Any]:
    """Parse a candidate table and return graph-ready data summary fields."""
    case_path = Path(data_file)
    table = parse_candidate_file(case_path)
    return {
        "case_path": str(case_path),
        "data_summary": summarize_candidate_table(table),
    }


def load_experiment_history(*, experiments_dir: str | Path) -> dict[str, Any]:
    """Load append-only experiment history and summarize local evidence."""
    store = ExperimentStore(root_dir=Path(experiments_dir))
    records = store.load_records()
    history_summary = summarize_local_history(records)
    evidence_profile = history_summary.get("evidence_profile", {})
    return {
        "experiment_records": records,
        "local_history_summary": history_summary,
        "evidence_profile": evidence_profile
        if isinstance(evidence_profile, dict)
        else {},
    }


def get_strategy_catalog() -> list[dict[str, Any]]:
    """Return executable strategy catalog entries available to the agent."""
    return strategy_space_catalog()


def get_strategy_primitive_schema() -> dict[str, Any]:
    """Return primitive schema for inline strategy configurations."""
    return strategy_primitive_schema()


def run_strategy_evaluation(
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
    selected_config_ids: list[str] | None = None,
    inline_configs: list[dict[str, Any]] | None = None,
    evidence_profile: dict[str, Any] | None = None,
    case_suite: CaseSuiteName = "robust",
) -> LocalEvaluationResult:
    """Materialize, evaluate, rank, and persist a strategy-space sweep."""
    return evaluate_strategy_space(
        search_space=search_space,
        data_file=data_file,
        baseline_solver=baseline_solver,
        experiments_dir=experiments_dir,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        candidate_limit=candidate_limit,
        provenance=provenance,
        hypothesis=hypothesis,
        selected_config_ids=selected_config_ids,
        inline_configs=inline_configs,
        evidence_profile=evidence_profile,
        case_suite=case_suite,
    )


def run_stop_search_evaluation(
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
    return evaluate_stop_search_candidate(
        solver_path=solver_path,
        data_file=data_file,
        baseline_solver=baseline_solver,
        experiments_dir=experiments_dir,
        timeout_seconds=timeout_seconds,
        provenance=provenance,
        evidence_profile=evidence_profile,
        case_suite=case_suite,
    )


def export_solver_candidate(*, source_path: str, destination_path: str) -> dict[str, str]:
    """Copy a standalone solver candidate into the configured solver path."""
    exported_path = package_solver(source_path, destination_path)
    return {
        "exported_solver_path": exported_path,
        "exported_solver_source_path": source_path,
    }


def write_research_report(
    *,
    context: Context,
    report_path: str | Path,
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
    notes: list[str] | None = None,
    llm_recommendations: list[str] | None = None,
    llm_error: str = "",
) -> dict[str, str]:
    """Render and persist the deterministic research report artifact."""
    _ = context
    report = render_research_report(
        research_goal=research_goal,
        case_path=case_path,
        solver_path=solver_path,
        data_summary=data_summary,
        latest_experiment=latest_experiment,
        baseline_suite=baseline_suite,
        baseline_full_case=baseline_full_case,
        variant_results=variant_results,
        best_variant_path=best_variant_path,
        best_variant_metrics=best_variant_metrics,
        local_history_summary=local_history_summary,
        evidence_profile=evidence_profile,
        research_decision=research_decision,
        decision_source=decision_source,
        llm_decision_raw=llm_decision_raw,
        llm_decision_attempts=llm_decision_attempts,
        decision_validation_errors=decision_validation_errors,
        decision_evidence=decision_evidence,
        autopilot_prior=autopilot_prior,
        solver_candidates=solver_candidates,
        exported_solver_path=exported_solver_path,
        exported_solver_source_path=exported_solver_source_path,
        notes=notes or [],
        llm_recommendations=llm_recommendations or [],
        llm_error=llm_error,
    )
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return {
        "report": report,
        "report_path": str(destination),
    }


def invoke_tool(name: str, **kwargs: Any) -> Any:
    """Invoke one registered tool by name."""
    try:
        tool = TOOL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown agent tool: {name}") from exc
    return tool(**kwargs)


def tool_schemas() -> list[dict[str, Any]]:
    """Return JSON-schema descriptors for all registered local tools."""
    return [tool.as_openai_tool() for tool in TOOLS]


TOOLS: list[AgentTool] = [
    AgentTool(
        name="summarize_candidate_data",
        description="Parse a local delivery candidate table and summarize rows, tasks, couriers, and bundle ratio.",
        args_schema={
            "type": "object",
            "properties": {"data_file": {"type": "string"}},
            "required": ["data_file"],
            "additionalProperties": False,
        },
        invoke=summarize_candidate_data,
    ),
    AgentTool(
        name="load_experiment_history",
        description="Load append-only local experiment records and summarize evidence for decision making.",
        args_schema={
            "type": "object",
            "properties": {"experiments_dir": {"type": "string"}},
            "required": ["experiments_dir"],
            "additionalProperties": False,
        },
        invoke=load_experiment_history,
    ),
    AgentTool(
        name="get_strategy_catalog",
        description="Return the executable strategy catalog available for LLM research decisions.",
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        invoke=get_strategy_catalog,
    ),
    AgentTool(
        name="get_strategy_primitive_schema",
        description="Return primitive schema for validating inline StrategyConfig decisions.",
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        invoke=get_strategy_primitive_schema,
    ),
    AgentTool(
        name="run_strategy_evaluation",
        description="Materialize configured solver variants, evaluate them on the local suite, rank them, and append evidence.",
        args_schema={
            "type": "object",
            "properties": {
                "search_space": {"type": "string"},
                "data_file": {"type": "string"},
                "baseline_solver": {"type": "string"},
                "experiments_dir": {"type": "string"},
                "output_dir": {"type": ["string", "null"]},
                "timeout_seconds": {"type": "number"},
                "candidate_limit": {"type": "integer", "minimum": 1},
                "provenance": {"type": ["object", "null"]},
                "hypothesis": {"type": "string"},
                "selected_config_ids": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "inline_configs": {
                    "type": ["array", "null"],
                    "items": {"type": "object"},
                },
                "evidence_profile": {"type": ["object", "null"]},
                "case_suite": {"type": "string"},
            },
            "required": ["search_space", "data_file", "baseline_solver", "experiments_dir"],
            "additionalProperties": False,
        },
        invoke=run_strategy_evaluation,
    ),
    AgentTool(
        name="run_stop_search_evaluation",
        description="Re-run a recommended solver candidate as local stop-search evidence.",
        args_schema={
            "type": "object",
            "properties": {
                "solver_path": {"type": "string"},
                "data_file": {"type": "string"},
                "baseline_solver": {"type": "string"},
                "experiments_dir": {"type": "string"},
                "timeout_seconds": {"type": "number"},
                "provenance": {"type": ["object", "null"]},
                "evidence_profile": {"type": ["object", "null"]},
                "case_suite": {"type": "string"},
            },
            "required": ["solver_path", "data_file", "baseline_solver", "experiments_dir"],
            "additionalProperties": False,
        },
        invoke=run_stop_search_evaluation,
    ),
    AgentTool(
        name="export_solver_candidate",
        description="Copy the selected standalone solver candidate into the configured solver output path.",
        args_schema={
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "destination_path": {"type": "string"},
            },
            "required": ["source_path", "destination_path"],
            "additionalProperties": False,
        },
        invoke=export_solver_candidate,
    ),
    AgentTool(
        name="write_research_report",
        description="Render and persist the deterministic AutoResearch report from graph state.",
        args_schema={
            "type": "object",
            "properties": {
                "report_path": {"type": "string"},
                "research_goal": {"type": "string"},
                "case_path": {"type": "string"},
                "solver_path": {"type": "string"},
            },
            "required": ["report_path", "research_goal", "case_path", "solver_path"],
            "additionalProperties": True,
        },
        invoke=write_research_report,
    ),
]

TOOL_REGISTRY: dict[str, AgentTool] = {tool.name: tool for tool in TOOLS}

