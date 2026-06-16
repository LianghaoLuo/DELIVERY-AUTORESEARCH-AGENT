"""Capture a sanitized replay fixture from a live AutoResearch run."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from autoresearch_agent.context import Context
from autoresearch_agent.graph import graph
from autoresearch_agent.replay import build_replay_fixture
from autoresearch_agent.state import InputState


async def main() -> None:
    """Run live iterations and write a compact replay fixture."""
    load_dotenv()
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sandbox_dir = Path(args.sandbox_dir)
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    states = await _run_capture_iterations(
        context=Context(
            experiments_dir=str(sandbox_dir / "experiments"),
            report_path=str(sandbox_dir / "report.md"),
            solver_entrypoint=args.solver_entrypoint,
            suggested_solver_output_path=str(sandbox_dir / "agent_suggested_solver.py"),
            enable_llm_decisions=True,
            enable_llm_recommendations=args.enable_llm_recommendations,
        ),
        research_goal=args.research_goal,
        max_iterations=args.max_iterations,
        stop_on_stop_search=not args.keep_running_after_stop,
    )
    capture_index = _select_capture_index(args.capture_iteration, len(states))
    final_state = states[capture_index]
    latest_experiment = final_state.get("latest_experiment", {})
    run_id = (
        str(latest_experiment.get("run_id", ""))
        if isinstance(latest_experiment, dict)
        else ""
    )
    fixture = build_replay_fixture(
        final_state=final_state,
        run_id=run_id,
        iteration=capture_index + 1,
        source="captured_llm_run",
    )
    _sanitize_capture_paths(fixture, sandbox_dir)
    output_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("wrote:", output_path)
    print("captured_iteration:", capture_index + 1, "of", len(states))
    print("solver_entrypoint:", args.solver_entrypoint)
    print("decision:", fixture["iterations"][0]["research_decision"]["search_space"])
    print("source:", fixture["iterations"][0]["decision_source"])


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture a no-key replay fixture from a live LLM run."
    )
    parser.add_argument(
        "--output",
        default="examples/sample_experiment_log.json",
        help="Destination JSON fixture path.",
    )
    parser.add_argument(
        "--sandbox-dir",
        default="/tmp/delivery_autoresearch_replay_capture",
        help="Temporary directory for generated reports, logs, and solver output.",
    )
    parser.add_argument(
        "--solver-entrypoint",
        default="solvers/solver.py",
        help="Baseline solver used for the captured run.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Number of live graph iterations to run before selecting a replay point.",
    )
    parser.add_argument(
        "--capture-iteration",
        default="middle",
        help=(
            "1-based iteration to store, or 'middle' to select the middle "
            "completed iteration."
        ),
    )
    parser.add_argument(
        "--research-goal",
        default=(
            "Capture a compact replay fixture from a multi-iteration "
            "LLM-guided AutoResearch run starting from the baseline solver."
        ),
    )
    parser.add_argument(
        "--enable-llm-recommendations",
        action="store_true",
        help="Also call the model for end-of-run recommendations.",
    )
    parser.add_argument(
        "--keep-running-after-stop",
        action="store_true",
        help="Continue capturing iterations even if the agent selects stop_search.",
    )
    return parser.parse_args()


async def _run_capture_iterations(
    *,
    context: Context,
    research_goal: str,
    max_iterations: int,
    stop_on_stop_search: bool,
) -> list[dict[str, Any]]:
    """Run live graph iterations and keep every completed state."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    states: list[dict[str, Any]] = []
    for _ in range(max_iterations):
        raw_result = await graph.ainvoke(
            InputState(research_goal=research_goal),
            context=context,
        )
        result = cast(dict[str, Any], raw_result)
        states.append(dict(result))
        decision = result.get("research_decision", {})
        search_space = (
            str(decision.get("search_space", ""))
            if isinstance(decision, dict)
            else ""
        )
        if stop_on_stop_search and search_space == "stop_search":
            break
    return states


def _select_capture_index(value: str, state_count: int) -> int:
    """Return a zero-based capture index from a user-facing selector."""
    if state_count < 1:
        raise ValueError("no completed states to capture")
    if value == "middle":
        return (state_count - 1) // 2
    try:
        selected = int(value)
    except ValueError as exc:
        raise ValueError("capture iteration must be an integer or 'middle'") from exc
    if selected < 1 or selected > state_count:
        raise ValueError(
            f"capture iteration must be between 1 and {state_count}; got {selected}"
        )
    return selected - 1


def _sanitize_capture_paths(fixture: dict[str, object], sandbox_dir: Path) -> None:
    """Replace local temporary capture paths with stable replay labels."""
    sandbox_prefix = str(sandbox_dir)
    sanitized = _sanitize_value(fixture, sandbox_prefix)
    fixture.clear()
    if isinstance(sanitized, dict):
        fixture.update(sanitized)


def _sanitize_value(value: object, sandbox_prefix: str) -> object:
    """Recursively replace sandbox paths in JSON-compatible values."""
    if isinstance(value, str):
        return value.replace(
            f"{sandbox_prefix}/agent_suggested_solver.py",
            "replay_artifacts/agent_suggested_solver.py",
        ).replace(sandbox_prefix, "replay_artifacts")
    if isinstance(value, list):
        return [_sanitize_value(item, sandbox_prefix) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item, sandbox_prefix)
            for key, item in value.items()
        }
    return value


if __name__ == "__main__":
    asyncio.run(main())
