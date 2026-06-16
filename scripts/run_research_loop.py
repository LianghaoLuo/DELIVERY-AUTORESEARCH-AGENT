"""Run the bounded local-only AutoResearch agent loop."""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

from autoresearch_agent.context import Context
from autoresearch_agent.research.loop_runner import run_agent_loop

DEFAULT_AGENT_EXPERIMENTS_DIR = "experiments/agent_runs/05271600"
DEFAULT_AGENT_REPORT_PATH = "reports/05271600.md"
DEFAULT_MAX_ITERATIONS = 5


async def main() -> None:
    """Invoke the local AutoResearch graph until stop evidence or max iterations."""
    load_dotenv()
    args = _parse_args()
    experiments_dir = os.environ.get(
        "AGENT_EXPERIMENTS_DIR",
        DEFAULT_AGENT_EXPERIMENTS_DIR,
    )
    report_path = os.environ.get("AGENT_REPORT_PATH", DEFAULT_AGENT_REPORT_PATH)
    max_iterations = int(
        os.environ.get("AGENT_MAX_ITERATIONS", str(args.max_iterations))
    )
    loop_result = await run_agent_loop(
        research_goal=args.research_goal,
        context=Context(
            experiments_dir=experiments_dir,
            report_path=report_path,
        ),
        max_iterations=max_iterations,
    )
    print("experiments_dir:", experiments_dir)
    print("iterations:", len(loop_result.iterations))
    print("stopped:", loop_result.stopped)
    for iteration in loop_result.iterations:
        print(
            "iteration:",
            iteration.iteration,
            "run_id:",
            iteration.run_id,
            "decision:",
            iteration.decision,
            "mean_proxy_score:",
            iteration.mean_proxy_score,
        )
    final_state = loop_result.final_state
    print("report_path:", final_state.get("report_path", ""))
    print("best_solver_path:", final_state.get("best_solver_path", ""))
    print("exported_solver_path:", final_state.get("exported_solver_path", ""))
    print("llm_recommendations:", len(final_state.get("llm_recommendations", [])))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the local-only AutoResearch agent loop."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=(
            "Maximum graph iterations to run. Can also be set with "
            "AGENT_MAX_ITERATIONS."
        ),
    )
    parser.add_argument(
        "--research-goal",
        default="Run the local baseline AutoResearch loop.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
