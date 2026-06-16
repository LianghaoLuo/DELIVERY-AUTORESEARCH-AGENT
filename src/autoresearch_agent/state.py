"""State objects for the offline AutoResearch workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class InputState:
    """External input accepted by the research graph."""

    research_goal: str = ""
    data_files: Sequence[str] = field(default_factory=list)


@dataclass
class State(InputState):
    """Complete graph state shared by planning, experiment, and report nodes."""

    notes: list[str] = field(default_factory=list)
    experiment_log: list[str] = field(default_factory=list)
    case_path: str = ""
    solver_path: str = ""
    data_summary: dict[str, int | float] = field(default_factory=dict)
    latest_experiment: dict[str, Any] = field(default_factory=dict)
    experiment_records: list[dict[str, Any]] = field(default_factory=list)
    local_history_summary: dict[str, Any] = field(default_factory=dict)
    evidence_profile: dict[str, Any] = field(default_factory=dict)
    research_decision: dict[str, Any] = field(default_factory=dict)
    decision_source: str = ""
    decision_evidence: list[str] = field(default_factory=list)
    llm_decision_raw: str = ""
    llm_decision_attempts: list[dict[str, Any]] = field(default_factory=list)
    decision_validation_errors: list[str] = field(default_factory=list)
    autopilot_prior: dict[str, Any] = field(default_factory=dict)
    strategy_catalog: list[dict[str, Any]] = field(default_factory=list)
    auto_sweep_results: list[dict[str, Any]] = field(default_factory=list)
    solver_candidates: list[dict[str, Any]] = field(default_factory=list)
    baseline_suite: dict[str, Any] = field(default_factory=dict)
    baseline_full_case: dict[str, Any] = field(default_factory=dict)
    variant_results: list[dict[str, Any]] = field(default_factory=list)
    best_variant_path: str = ""
    best_variant_metrics: dict[str, Any] = field(default_factory=dict)
    exported_solver_path: str = ""
    exported_solver_source_path: str = ""
    llm_recommendations: list[str] = field(default_factory=list)
    llm_recommendation_text: str = ""
    llm_error: str = ""
    best_solver_path: str = ""
    report: str = ""
    report_path: str = ""
