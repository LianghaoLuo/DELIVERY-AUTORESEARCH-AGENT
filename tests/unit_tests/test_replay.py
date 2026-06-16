import json
from pathlib import Path

import pytest

from autoresearch_agent.replay import (
    build_replay_fixture,
    load_replay_fixture,
    render_replay_markdown,
)


def test_build_replay_fixture_and_render_markdown() -> None:
    fixture = build_replay_fixture(
        final_state={
            "research_goal": "Replay one LLM-guided iteration.",
            "case_path": "data/large_seed301.txt",
            "solver_path": "solvers/prior_solver.py",
            "data_summary": {
                "candidate_count": 33780,
                "task_count": 40,
                "courier_count": 80,
                "bundled_candidate_ratio": 0.9,
            },
            "local_history_summary": {
                "record_count": 0,
                "evidence_profile": {"failure_mode": "bootstrap"},
            },
            "evidence_profile": {
                "failure_mode": "bootstrap",
                "strategy_family": "broad_strategy",
                "evidence_score": 1.0,
            },
            "decision_source": "llm",
            "research_decision": {
                "search_space": "broad_strategy",
                "hypothesis": "Probe broad strategy configs.",
                "config_ids": ["broad_alpha90"],
                "candidate_limit": 1,
                "failure_mode": "bootstrap",
            },
            "llm_decision_attempts": [
                {"attempt": 1, "is_valid": True, "validation_errors": []}
            ],
            "baseline_suite": {
                "aggregate_metrics": {
                    "is_valid": True,
                    "case_count": 10,
                    "mean_proxy_score": 100.0,
                }
            },
            "best_variant_path": "experiments/generated_variants/sample.py",
            "best_variant_metrics": {"mean_proxy_score": 123.0},
            "solver_candidates": [
                {
                    "rank": 1,
                    "variant_path": "experiments/generated_variants/sample.py",
                    "family": "broad_strategy",
                    "pipeline": "greedy",
                }
            ],
            "latest_experiment": {"run_id": "run-1"},
        },
        source="test",
    )

    rendered = render_replay_markdown(fixture)

    assert fixture["schema_version"] == 1
    assert "AutoResearch Replay" in rendered
    assert "Iteration 1" in rendered
    assert "Source: `llm`" in rendered
    assert "Search space: `broad_strategy`" in rendered
    assert "Baseline cases: `10`" in rendered


def test_load_replay_fixture_validates_shape(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps({"schema_version": 1, "iterations": [{"iteration": 1}]}),
        encoding="utf-8",
    )

    loaded = load_replay_fixture(path)

    assert loaded["schema_version"] == 1


def test_sample_replay_fixture_renders_without_local_tmp_paths() -> None:
    fixture = load_replay_fixture("examples/sample_experiment_log.json")
    rendered = render_replay_markdown(fixture)

    assert "Source: `llm`" in rendered
    assert "Solver baseline: `solvers/solver.py`" in rendered
    assert "Search space: `bundle_merge_duplicate`" in rendered
    assert "/tmp/" not in rendered


def test_load_replay_fixture_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps({"schema_version": 999, "iterations": [{"iteration": 1}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported replay schema"):
        load_replay_fixture(path)
