"""Run a lightweight tool-calling demo over the local agent tool registry."""

from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv

from autoresearch_agent.context import Context
from autoresearch_agent.tool_calling_demo import (
    run_llm_tool_calling_demo,
    run_replay_tool_calling_demo,
)


async def main() -> None:
    """Run the demo in replay mode or with the configured LLM."""
    load_dotenv()
    args = _parse_args()
    if args.use_llm:
        result = await run_llm_tool_calling_demo(
            context=Context(),
            data_file=args.data_file,
            experiments_dir=args.experiments_dir,
        )
    else:
        result = run_replay_tool_calling_demo(
            data_file=args.data_file,
            experiments_dir=args.experiments_dir,
        )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate the typed local tool registry. By default this runs a "
            "deterministic no-key replay; pass --use-llm to ask the configured "
            "model to emit tool calls."
        )
    )
    parser.add_argument(
        "--data-file",
        default="data/large_seed301.txt",
        help="Candidate table used by summarize_candidate_data.",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Experiment history directory used by load_experiment_history.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the configured LLM to produce tool calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
