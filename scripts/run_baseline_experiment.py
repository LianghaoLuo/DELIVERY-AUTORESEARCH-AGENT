"""Run the baseline solver once and persist an experiment record."""

from __future__ import annotations

from autoresearch_agent.research.experiment_store import ExperimentStore
from autoresearch_agent.solver_dev.runner import run_solver_case


def main() -> None:
    """Run the current prior solver on the provided large seed case."""
    result = run_solver_case(
        "solvers/prior_solver.py",
        "data/large_seed301.txt",
        timeout_seconds=10.0,
    )
    record = ExperimentStore().append_solver_run(
        result,
        label="prior-solver-large-seed301",
        notes="Current robust-proxy prior solver on the local seed case.",
    )
    print(record["run_id"])


if __name__ == "__main__":
    main()
