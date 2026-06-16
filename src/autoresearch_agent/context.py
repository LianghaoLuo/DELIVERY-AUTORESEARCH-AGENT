"""Runtime configuration for the AutoResearch agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from autoresearch_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Agent-side configuration.

    This context is for Python 3.10/3.11+ research code only. The final
    ``solvers/solver.py`` must remain independent from these settings and
    from all LangChain/LangGraph dependencies.
    """

    system_prompt: str = field(default=prompts.SYSTEM_PROMPT)
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="deepseek/deepseek-chat"
    )
    model_base_url: str = field(default="https://api.deepseek.com")
    model_api_key_env: str = field(default="DEEPSEEK_API_KEY")
    enable_llm_decisions: bool = field(default=True)
    enable_llm_recommendations: bool = field(default=True)
    research_goal: str = field(
        default="Build an offline AutoResearch Agent for the delivery assignment problem."
    )
    data_dir: str = field(default="data")
    solvers_dir: str = field(default="solvers")
    experiments_dir: str = field(default="experiments")
    solver_entrypoint: str = field(default="solvers/prior_solver.py")
    local_case_suite: str = field(default="robust")
    suggested_solver_output_path: str = field(
        default="solvers/agent_suggested_solver.py"
    )
    report_path: str = field(default="reports/autoresearch_report.md")
    case_timeout_seconds: int = field(default=10)

    def __post_init__(self) -> None:
        """Allow environment variables to override default context values."""
        for f in fields(self):
            if not f.init:
                continue
            if getattr(self, f.name) == f.default:
                raw_value = os.environ.get(f.name.upper())
                if raw_value is None:
                    continue
                if isinstance(f.default, bool):
                    setattr(self, f.name, raw_value.lower() in {"1", "true", "yes", "on"})
                elif isinstance(f.default, int):
                    setattr(self, f.name, int(raw_value))
                else:
                    setattr(self, f.name, raw_value)
