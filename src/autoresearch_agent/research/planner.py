"""Initial strategy-plan text for delivery assignment research."""

from __future__ import annotations


def propose_strategy_plan() -> list[str]:
    """Return the local-only AutoResearch loop plan."""
    return [
        "Inspect candidate-table statistics and build the configured local case suite.",
        "Load append-only local experiment history from the current agent session.",
        "Infer the dominant local failure mode from proxy metrics, validity, and worst-case evidence.",
        "Run the LLM-selected catalog sweep or stop-search evidence on local cases only.",
        "Export the best local candidate to agent_suggested_solver.py and render a report.",
        "Keep external score chasing outside the agent loop; use robust local evidence as the search signal.",
    ]
