"""Interactively write local DeepSeek settings to .env."""

from __future__ import annotations

from getpass import getpass
from pathlib import Path

ENV_PATH = Path(".env")


def main() -> None:
    """Prompt for a DeepSeek key and write a local .env file."""
    api_key = getpass("DeepSeek API key: ").strip()
    if not api_key:
        raise SystemExit("No API key provided.")

    lines = [
        "DEEPSEEK_API_KEY=" + api_key,
        "MODEL=deepseek/deepseek-chat",
        "MODEL_BASE_URL=https://api.deepseek.com",
        "MODEL_API_KEY_ENV=DEEPSEEK_API_KEY",
        "ENABLE_LLM_DECISIONS=true",
        "ENABLE_LLM_RECOMMENDATIONS=true",
        "LANGSMITH_PROJECT=delivery-autoresearch-agent",
        "DATA_DIR=data",
        "SOLVERS_DIR=solvers",
        "EXPERIMENTS_DIR=experiments",
        "SOLVER_ENTRYPOINT=solvers/prior_solver.py",
        "REPORT_PATH=reports/autoresearch_report.md",
        "CASE_TIMEOUT_SECONDS=10",
    ]
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print("Wrote .env. This file is ignored by git.")


if __name__ == "__main__":
    main()
