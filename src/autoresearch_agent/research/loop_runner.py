"""Outer loop runner for the local-only AutoResearch agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from autoresearch_agent.context import Context
from autoresearch_agent.graph import graph
from autoresearch_agent.state import InputState


@dataclass(frozen=True)
class AgentLoopIteration:
    """Compact summary for one graph invocation in the agent loop."""

    iteration: int
    run_id: str
    decision: str
    best_solver_path: str
    exported_solver_path: str
    report_path: str
    mean_proxy_score: float
    stopped: bool


@dataclass(frozen=True)
class AgentLoopResult:
    """Result from running one bounded local AutoResearch loop."""

    iterations: list[AgentLoopIteration]
    final_state: dict[str, Any]

    @property
    def stopped(self) -> bool:
        """Return whether the loop reached local stop-search evidence."""
        return bool(self.iterations and self.iterations[-1].stopped)


async def run_agent_loop(
    *,
    context: Context,
    research_goal: str,
    max_iterations: int,
    stop_on_stop_search: bool = True,
) -> AgentLoopResult:
    """Run repeated local graph iterations until stop evidence or max iterations.

    The graph and this runner intentionally use local experiment history only.
    External benchmark score records are outside the agent runtime.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    summaries: list[AgentLoopIteration] = []
    final_state: dict[str, Any] = {}
    for iteration in range(1, max_iterations + 1):
        raw_result = await graph.ainvoke(
            InputState(research_goal=research_goal),
            context=context,
        )
        result = cast(dict[str, Any], raw_result)
        final_state = dict(result)
        decision = str(result.get("research_decision", {}).get("search_space", ""))
        latest = result.get("latest_experiment", {})
        metrics = result.get("best_variant_metrics", {})
        stopped = decision == "stop_search"
        summaries.append(
            AgentLoopIteration(
                iteration=iteration,
                run_id=str(latest.get("run_id", "")),
                decision=decision,
                best_solver_path=str(result.get("best_solver_path", "")),
                exported_solver_path=str(result.get("exported_solver_path", "")),
                report_path=str(result.get("report_path", "")),
                mean_proxy_score=float(metrics.get("mean_proxy_score", 0.0)),
                stopped=stopped,
            )
        )
        if stop_on_stop_search and stopped:
            break

    return AgentLoopResult(iterations=summaries, final_state=final_state)
