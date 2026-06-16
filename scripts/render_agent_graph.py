"""Render Mermaid diagrams for the AutoResearch agent README."""

from __future__ import annotations

OUTER_LOOP_MERMAID = """flowchart TD
    A([Start bounded AutoResearch loop]) --> B[Invoke LangGraph iteration]
    B --> C[Append experiment evidence]
    C --> D{Decision is stop_search?}
    D -- yes --> E([Stop with local evidence])
    D -- no --> F{Max iterations reached?}
    F -- no --> B
    F -- yes --> G([Stop at iteration budget])
"""

GRAPH_ITERATION_MERMAID = """flowchart LR
    Start([__start__]) --> Plan[plan_research]
    Plan --> History[load_local_history]
    History --> Decide[decide_next_experiment_node]
    Decide --> Run[run_selected_experiment]
    Run --> Export[export_suggested_solver]
    Export --> Reflect[generate_research_suggestions]
    Reflect --> Report[write_report]
    Report --> End([__end__])

    Data[(case data)] --> Plan
    Store[(experiment history)] --> History
    Store --> Decide
    Catalog[(strategy catalog)] --> Decide
    LLM{{LLM structured decision}} --> Decide
    Decide --> Guardrail{validate / repair}
    Guardrail -- valid --> Run
    Guardrail -- invalid --> Fail[[fail closed]]
    Run --> Variants[(generated solver variants)]
    Run --> Store
    Export --> Suggested[(agent_suggested_solver.py)]
    Report --> Reports[(research report)]
"""


def main() -> None:
    """Print README-ready Mermaid diagrams."""
    print("## Outer loop")
    print()
    print("```mermaid")
    print(OUTER_LOOP_MERMAID.strip())
    print("```")
    print()
    print("## Single graph iteration")
    print()
    print("```mermaid")
    print(GRAPH_ITERATION_MERMAID.strip())
    print("```")


if __name__ == "__main__":
    main()
