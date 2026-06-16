import importlib
import json
from pathlib import Path

import pytest

from autoresearch_agent.context import Context
from autoresearch_agent.graph import graph
from autoresearch_agent.research import loop_runner as loop_runner_module
from autoresearch_agent.research.loop_runner import run_agent_loop
from autoresearch_agent.research.planner import propose_strategy_plan
from autoresearch_agent.research.strategy_space import build_strategy_configs
from autoresearch_agent.tools import TOOL_REGISTRY, TOOLS, invoke_tool, tool_schemas

pytestmark = pytest.mark.anyio
graph_module = importlib.import_module("autoresearch_agent.graph")


def test_context_defaults_match_solver_layout() -> None:
    context = Context()

    assert context.model == "deepseek/deepseek-chat"
    assert context.model_base_url == "https://api.deepseek.com"
    assert context.model_api_key_env == "DEEPSEEK_API_KEY"
    assert context.enable_llm_decisions
    assert context.enable_llm_recommendations
    assert context.data_dir == "data"
    assert context.solvers_dir == "solvers"
    assert context.solver_entrypoint == "solvers/prior_solver.py"
    assert context.local_case_suite == "robust"
    assert context.suggested_solver_output_path == "solvers/agent_suggested_solver.py"
    assert context.report_path == "reports/autoresearch_report.md"
    assert context.case_timeout_seconds == 10


def test_graph_is_compiled() -> None:
    assert graph.name == "Delivery AutoResearch Agent"


def test_strategy_plan_describes_local_only_agent_loop() -> None:
    plan = propose_strategy_plan()

    assert plan
    assert not any(item.startswith("TODO:") for item in plan)
    assert any("local" in item.lower() for item in plan)
    assert any("robust local evidence" in item for item in plan)


def test_tool_registry_exposes_local_agent_capabilities() -> None:
    tool_names = {tool.name for tool in TOOLS}

    assert len(tool_names) == len(TOOLS)
    assert {
        "summarize_candidate_data",
        "load_experiment_history",
        "get_strategy_catalog",
        "get_strategy_primitive_schema",
        "run_strategy_evaluation",
        "run_stop_search_evaluation",
        "export_solver_candidate",
        "write_research_report",
    }.issubset(tool_names)
    assert TOOL_REGISTRY["summarize_candidate_data"].args_schema["type"] == "object"
    assert tool_schemas()[0]["type"] == "function"
    with pytest.raises(ValueError, match="unknown agent tool"):
        invoke_tool("missing_tool")


async def test_graph_runs_local_research_loop(tmp_path) -> None:
    experiments_dir = tmp_path / "experiments"
    _write_diminishing_duplicate_history(experiments_dir)
    context = Context(
        experiments_dir=str(experiments_dir),
        report_path=str(tmp_path / "report.md"),
        suggested_solver_output_path=str(tmp_path / "agent_suggested_solver.py"),
        enable_llm_decisions=False,
        enable_llm_recommendations=False,
    )

    result = await graph.ainvoke(
        {"research_goal": "smoke test local autoresearch loop"},
        context=context,
    )

    assert result["data_summary"]["candidate_count"] == 33780
    assert result["baseline_suite"]["aggregate_metrics"]["is_valid"]
    assert result["baseline_suite"]["aggregate_metrics"]["case_count"] == 10
    assert result["baseline_full_case"]["metrics"]["is_valid"]
    assert result["baseline_full_case"]["metrics"]["assigned_task_count"] == 40
    assert result["local_history_summary"]["duplicate_augment_diminishing_returns"]
    assert result["evidence_profile"]["failure_mode"] == "low_expected_success"
    assert result["research_decision"]["search_space"] == "risk_tier_duplicate"
    assert result["latest_experiment"]["provenance"]["origin"] == "agent_loop"
    assert (
        result["latest_experiment"]["provenance"]["agent_version"] == "strategy_loop_v3"
    )
    assert result["decision_source"] == "local_autopilot"
    assert (
        result["latest_experiment"]["provenance"]["selected_by"]
        == "local_autopilot"
    )
    assert result["latest_experiment"]["provenance"]["accepted_source"] == (
        "local_autopilot"
    )
    assert result["decision_validation_errors"] == []
    assert result["autopilot_prior"]
    assert result["strategy_catalog"]
    assert result["decision_evidence"]
    assert result["solver_candidates"]
    assert len(result["variant_results"]) >= 1
    assert result["best_variant_path"]
    assert "mean_proxy_score" in result["best_variant_metrics"]
    assert result["best_solver_path"] == result["best_variant_path"]
    assert result["exported_solver_path"] == str(tmp_path / "agent_suggested_solver.py")
    assert result["exported_solver_source_path"] == result["best_variant_path"]
    assert (tmp_path / "agent_suggested_solver.py").exists()
    assert "AutoResearch Report" in result["report"]
    assert "Research Decision" in result["report"]
    assert "Decision source: `local_autopilot`" in result["report"]
    assert "Failure Mode Evidence" in result["report"]
    assert "Decision Guardrail" in result["report"]
    assert "Local Autopilot Plan" in result["report"]
    assert "Baseline Suite Experiment" in result["report"]
    assert "Evaluation Boundary" in result["report"]
    assert "Variant Suite Leaderboard" in result["report"]
    assert "Recommended Candidates" in result["report"]
    assert "Exported solver" in result["report"]
    assert result["llm_recommendations"]
    assert result["llm_recommendations"][0].startswith("LLM disabled:")
    assert (experiments_dir / "runs.jsonl").exists()
    assert (tmp_path / "report.md").exists()


async def test_graph_uses_valid_llm_research_decision(tmp_path, monkeypatch) -> None:
    selected_search_spaces: list[str] = []
    config_id = _seed_config_id("bundle_merge_duplicate", "explore_alpha92p5_m010")

    class FakeMessage:
        content = json.dumps(
            {
                "search_space": "bundle_merge_duplicate",
                "hypothesis": "Probe bundle merge under scarce courier pressure.",
                "config_ids": [config_id],
                "configs": [],
                "candidate_limit": 1,
                "expected_evidence": ["mean proxy on robust suite"],
                "failure_mode": "scarce_courier_pressure",
                "strategy_family": "bundle_merge_duplicate",
                "reasons": ["local history shows duplicate searches diminishing"],
            }
        )

    class FakeModel:
        async def ainvoke(self, messages):
            assert "Executable strategy catalog" in messages[-1]["content"]
            assert "Autopilot prior decision" not in messages[-1]["content"]
            return FakeMessage()

    def fake_evaluate_strategy_space(**kwargs):
        selected_search_spaces.append(kwargs["search_space"])
        return _fake_strategy_evaluation(tmp_path, provenance=kwargs["provenance"])

    monkeypatch.setattr(
        graph_module,
        "load_chat_model",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        graph_module,
        "run_strategy_evaluation",
        fake_evaluate_strategy_space,
    )
    experiments_dir = tmp_path / "experiments"
    _write_diminishing_duplicate_history(experiments_dir)
    context = Context(
        experiments_dir=str(experiments_dir),
        report_path=str(tmp_path / "report.md"),
        suggested_solver_output_path=str(tmp_path / "agent_suggested_solver.py"),
        enable_llm_decisions=True,
        enable_llm_recommendations=False,
    )

    result = await graph.ainvoke(
        {"research_goal": "smoke test llm decision loop"},
        context=context,
    )

    assert selected_search_spaces == ["bundle_merge_duplicate"]
    assert result["decision_source"] == "llm"
    assert result["research_decision"]["search_space"] == "bundle_merge_duplicate"
    assert result["research_decision"]["candidate_limit"] == 1
    assert result["research_decision"]["config_ids"] == [config_id]
    assert result["decision_validation_errors"] == []
    assert result["llm_decision_attempts"][0]["is_valid"]
    assert "bundle_merge_duplicate" in result["llm_decision_raw"]
    assert result["latest_experiment"]["provenance"]["selected_by"] == "llm"
    assert result["latest_experiment"]["provenance"]["accepted_source"] == "llm"
    assert "Decision source: `llm`" in result["report"]
    assert "LLM Decision Diagnostics" in result["report"]


async def test_graph_repairs_invalid_llm_research_decision(
    tmp_path, monkeypatch
) -> None:
    selected_search_spaces: list[str] = []
    config_id = _seed_config_id("bundle_merge_duplicate", "explore_alpha92p5_m010")

    class FakeModel:
        def __init__(self) -> None:
            self.call_count = 0

        async def ainvoke(self, messages):
            self.call_count += 1
            if self.call_count == 1:
                assert "Autopilot prior decision" not in messages[-1]["content"]
                return _FakeMessage(
                    json.dumps(
                        {
                            "search_space": "bundle_merge_duplicate",
                            "hypothesis": "Old selector shape.",
                            "selected_profiles": ["explore_alpha92p5_m010"],
                            "candidate_limit": 1,
                            "expected_evidence": [],
                            "failure_mode": "scarce_courier_pressure",
                            "strategy_family": "bundle_merge_duplicate",
                        }
                    )
                )
            assert "Allowed values for repair" in messages[-1]["content"]
            assert "Autopilot prior decision" not in messages[-1]["content"]
            return _FakeMessage(
                json.dumps(
                    {
                        "search_space": "bundle_merge_duplicate",
                        "hypothesis": "Repair to a concrete config.",
                        "config_ids": [config_id],
                        "configs": [],
                        "candidate_limit": 1,
                        "expected_evidence": ["mean proxy on robust suite"],
                        "failure_mode": "scarce_courier_pressure",
                        "strategy_family": "bundle_merge_duplicate",
                        "reasons": ["repair picked an allowed config"],
                    }
                )
            )

    fake_model = FakeModel()

    def fake_evaluate_strategy_space(**kwargs):
        selected_search_spaces.append(kwargs["search_space"])
        return _fake_strategy_evaluation(tmp_path, provenance=kwargs["provenance"])

    monkeypatch.setattr(
        graph_module,
        "load_chat_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        graph_module,
        "run_strategy_evaluation",
        fake_evaluate_strategy_space,
    )
    experiments_dir = tmp_path / "experiments"
    _write_diminishing_duplicate_history(experiments_dir)
    context = Context(
        experiments_dir=str(experiments_dir),
        report_path=str(tmp_path / "report.md"),
        suggested_solver_output_path=str(tmp_path / "agent_suggested_solver.py"),
        enable_llm_decisions=True,
        enable_llm_recommendations=False,
    )

    result = await graph.ainvoke(
        {"research_goal": "smoke test llm decision repair"},
        context=context,
    )

    assert selected_search_spaces == ["bundle_merge_duplicate"]
    assert fake_model.call_count == 2
    assert result["decision_source"] == "llm_repair"
    assert result["decision_validation_errors"] == []
    assert [attempt["is_valid"] for attempt in result["llm_decision_attempts"]] == [
        False,
        True,
    ]
    assert result["latest_experiment"]["provenance"]["accepted_source"] == (
        "llm_repair"
    )


async def test_graph_rejects_invalid_llm_research_decision(
    tmp_path, monkeypatch
) -> None:
    selected_search_spaces: list[str] = []

    class FakeMessage:
        content = "not json"

    class FakeModel:
        async def ainvoke(self, messages):
            assert messages
            assert "Autopilot prior decision" not in messages[-1]["content"]
            return FakeMessage()

    def fake_evaluate_strategy_space(**kwargs):
        selected_search_spaces.append(kwargs["search_space"])
        return _fake_strategy_evaluation(tmp_path, provenance=kwargs["provenance"])

    monkeypatch.setattr(
        graph_module,
        "load_chat_model",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        graph_module,
        "run_strategy_evaluation",
        fake_evaluate_strategy_space,
    )
    experiments_dir = tmp_path / "experiments"
    _write_diminishing_duplicate_history(experiments_dir)
    context = Context(
        experiments_dir=str(experiments_dir),
        report_path=str(tmp_path / "report.md"),
        suggested_solver_output_path=str(tmp_path / "agent_suggested_solver.py"),
        enable_llm_decisions=True,
        enable_llm_recommendations=False,
    )

    with pytest.raises(ValueError, match="decision guardrail"):
        await graph.ainvoke(
            {"research_goal": "smoke test invalid llm decision guardrail"},
            context=context,
        )

    assert selected_search_spaces == []


async def test_graph_uses_llm_for_research_recommendations(
    tmp_path, monkeypatch
) -> None:
    class FakeMessage:
        content = (
            "- Try willingness-adjusted greedy.\n"
            "- Compare bundle-first ordering.\n"
            "- Keep solver Python 3.6 compatible."
        )

    class FakeModel:
        async def ainvoke(self, messages):
            assert messages
            return FakeMessage()

    monkeypatch.setattr(
        graph_module,
        "load_chat_model",
        lambda *args, **kwargs: FakeModel(),
    )
    experiments_dir = tmp_path / "experiments"
    _write_diminishing_duplicate_history(experiments_dir)
    context = Context(
        experiments_dir=str(experiments_dir),
        report_path=str(tmp_path / "report.md"),
        suggested_solver_output_path=str(tmp_path / "agent_suggested_solver.py"),
        enable_llm_decisions=False,
        enable_llm_recommendations=True,
    )

    result = await graph.ainvoke(
        {"research_goal": "smoke test llm recommendation loop"},
        context=context,
    )

    assert result["llm_error"] == ""
    assert result["llm_recommendations"][0] == "Try willingness-adjusted greedy."
    assert result["variant_results"]
    assert "Try willingness-adjusted greedy." in result["report"]


async def test_outer_agent_loop_stops_on_stop_search(tmp_path, monkeypatch) -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.call_count = 0

        async def ainvoke(self, state, context):
            self.call_count += 1
            decision = "risk_tier_duplicate"
            if self.call_count == 2:
                decision = "stop_search"
            return {
                "latest_experiment": {"run_id": f"run-{self.call_count}"},
                "research_decision": {"search_space": decision},
                "best_solver_path": f"best-{self.call_count}.py",
                "exported_solver_path": f"export-{self.call_count}.py",
                "report_path": str(tmp_path / "report.md"),
                "best_variant_metrics": {"mean_proxy_score": -100.0},
                "llm_recommendations": [],
            }

    fake_graph = FakeGraph()
    monkeypatch.setattr(loop_runner_module, "graph", fake_graph)
    result = await run_agent_loop(
        context=Context(
            experiments_dir=str(tmp_path / "experiments"),
            report_path=str(tmp_path / "report.md"),
            enable_llm_decisions=False,
            enable_llm_recommendations=False,
        ),
        research_goal="test local loop",
        max_iterations=5,
    )

    assert fake_graph.call_count == 2
    assert result.stopped
    assert [item.decision for item in result.iterations] == [
        "risk_tier_duplicate",
        "stop_search",
    ]
    assert result.final_state["best_solver_path"] == "best-2.py"


def _write_diminishing_duplicate_history(experiments_dir: Path) -> None:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    best_path = (
        "experiments/generated_variants/p1c_local_improve/"
        "p1c_local_improve_a090_p001_t008p5.py"
    )
    records = [
        _variant_record("broad-strategy-sweep", "broad_best.py"),
        _variant_record("local-improve-strategy-sweep", best_path),
        _variant_record("local-improve-strategy-sweep", best_path),
        _variant_record("duplicate-augment-strategy-sweep", best_path),
        _variant_record("duplicate-augment-strategy-sweep", best_path),
        _variant_record("duplicate-augment-strategy-sweep", best_path),
    ]
    with (experiments_dir / "runs.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _variant_record(label: str, best_path: str) -> dict:
    return {
        "label": label,
        "best_variant_path": best_path,
        "batch": {
            "variant_results": [
                _variant_result(1, best_path, -100.0),
                _variant_result(2, "second.py", -95.0),
            ]
        },
    }


def _variant_result(rank: int, path: str, mean_proxy_score: float) -> dict:
    return {
        "rank": rank,
        "variant_path": path,
        "aggregate_metrics": {
            "is_valid": True,
            "case_count": 5,
            "timeout_count": 0,
            "invalid_case_count": 0,
            "mean_proxy_score": mean_proxy_score,
            "mean_expected_success_ratio": 0.82,
            "mean_task_coverage_ratio": 1.0,
            "worst_case_id": "low_willingness_stress_20_tasks",
        },
    }


class _FakeStrategyEvaluation:
    def __init__(self, variant_path: Path, provenance: dict) -> None:
        self.latest_record = {
            "run_id": "fake-run",
            "provenance": provenance,
        }
        self.baseline_suite = {
            "aggregate_metrics": {
                "is_valid": True,
                "case_count": 1,
                "invalid_case_count": 0,
                "timeout_count": 0,
                "mean_task_coverage_ratio": 1.0,
                "mean_expected_success_ratio": 0.9,
                "mean_proxy_score": -100.0,
                "worst_case_id": "fake",
            }
        }
        self.baseline_full_case = {
            "metrics": {
                "assigned_task_count": 1,
                "unassigned_task_count": 0,
                "total_score": 100.0,
                "expected_success_ratio": 0.9,
                "proxy_score": -100.0,
            },
            "validation": {"is_valid": True},
        }
        self.batch_payload = {
            "variant_results": [
                {
                    "rank": 1,
                    "variant_path": str(variant_path),
                    "aggregate_metrics": {
                        "is_valid": True,
                        "case_count": 1,
                        "timeout_count": 0,
                        "mean_task_coverage_ratio": 1.0,
                        "mean_expected_success_ratio": 0.9,
                        "mean_proxy_score": -101.0,
                        "worst_case_id": "fake",
                    },
                }
            ]
        }
        self.best_variant_path = str(variant_path)
        self.best_variant_metrics = {"mean_proxy_score": -101.0}
        self.leaderboard_rows = []
        self.solver_candidates = []


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


def _seed_config_id(search_space: str, profile: str) -> str:
    for config in build_strategy_configs(search_space):
        if profile in config.name or profile in config.tags:
            return config.config_id
    raise AssertionError(f"missing seed profile {profile}")


def _fake_strategy_evaluation(
    tmp_path: Path,
    *,
    provenance: dict,
) -> _FakeStrategyEvaluation:
    variant_path = tmp_path / "fake_variant.py"
    variant_path.write_text("def solve(input_text):\n    return ''\n")
    return _FakeStrategyEvaluation(variant_path, provenance)
