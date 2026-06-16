"""Small tool-calling demo built on the local agent tool registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.messages import BaseMessage

from autoresearch_agent.context import Context
from autoresearch_agent.tools import TOOLS, invoke_tool
from autoresearch_agent.utils import get_message_text, load_chat_model

DEMO_TOOL_NAMES: tuple[str, ...] = (
    "summarize_candidate_data",
    "load_experiment_history",
    "get_strategy_catalog",
)


@dataclass(frozen=True)
class DemoToolCall:
    """One local tool call proposed by replay data or by an LLM."""

    name: str
    args: dict[str, Any]
    source: str = "replay"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "args": self.args,
            "source": self.source,
        }


@dataclass(frozen=True)
class DemoToolResult:
    """Compact result from executing one demo tool call."""

    call: DemoToolCall
    preview: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "call": self.call.as_dict(),
            "preview": self.preview,
        }


@dataclass(frozen=True)
class ToolCallingDemoResult:
    """Complete payload from the tool-calling demo."""

    mode: str
    model_text: str
    tool_calls: list[DemoToolCall]
    tool_results: list[DemoToolResult]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "mode": self.mode,
            "model_text": self.model_text,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
            "tool_results": [result.as_dict() for result in self.tool_results],
        }


def demo_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI-compatible schemas for the safe demo tool subset."""
    allowed = set(DEMO_TOOL_NAMES)
    return [tool.as_openai_tool() for tool in TOOLS if tool.name in allowed]


def replay_tool_calls(
    *,
    data_file: str,
    experiments_dir: str,
) -> list[DemoToolCall]:
    """Return a deterministic no-key tool-call sequence for demos and tests."""
    return [
        DemoToolCall(
            name="summarize_candidate_data",
            args={"data_file": data_file},
        ),
        DemoToolCall(
            name="load_experiment_history",
            args={"experiments_dir": experiments_dir},
        ),
        DemoToolCall(
            name="get_strategy_catalog",
            args={},
        ),
    ]


def run_replay_tool_calling_demo(
    *,
    data_file: str,
    experiments_dir: str,
) -> ToolCallingDemoResult:
    """Run the deterministic no-key tool-calling demo."""
    calls = replay_tool_calls(data_file=data_file, experiments_dir=experiments_dir)
    return ToolCallingDemoResult(
        mode="replay",
        model_text="Replay mode: deterministic local tool calls, no LLM required.",
        tool_calls=calls,
        tool_results=execute_demo_tool_calls(calls),
    )


async def run_llm_tool_calling_demo(
    *,
    context: Context,
    data_file: str,
    experiments_dir: str,
) -> ToolCallingDemoResult:
    """Ask the configured model to choose local tools, then execute them."""
    model = load_chat_model(
        context.model,
        base_url=context.model_base_url,
        api_key_env=context.model_api_key_env,
    )
    tool_bound_model = model.bind_tools(demo_tool_schemas())
    response = cast(
        BaseMessage,
        await tool_bound_model.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are demonstrating tool calling for a local "
                        "AutoResearch project. Use only the provided tools."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Inspect this project by calling tools. First summarize "
                        f"candidate data at {data_file!r}, then load experiment "
                        f"history from {experiments_dir!r}, then inspect the "
                        "strategy catalog. Do not invent results."
                    ),
                },
            ]
        ),
    )
    calls = extract_tool_calls(response)
    if not calls:
        raise ValueError("model response did not include any tool calls")
    return ToolCallingDemoResult(
        mode="llm",
        model_text=get_message_text(response),
        tool_calls=calls,
        tool_results=execute_demo_tool_calls(calls),
    )


def extract_tool_calls(message: BaseMessage) -> list[DemoToolCall]:
    """Extract LangChain/OpenAI-style tool calls from a model message."""
    calls = _extract_langchain_tool_calls(message)
    if calls:
        return calls
    return _extract_openai_tool_calls(message)


def execute_demo_tool_calls(calls: list[DemoToolCall]) -> list[DemoToolResult]:
    """Execute allow-listed demo tool calls through the registry."""
    results: list[DemoToolResult] = []
    for call in calls:
        if call.name not in DEMO_TOOL_NAMES:
            raise ValueError(f"tool is not allowed in the demo: {call.name}")
        raw_result = invoke_tool(call.name, **call.args)
        results.append(
            DemoToolResult(
                call=call,
                preview=_preview_tool_result(call.name, raw_result),
            )
        )
    return results


def _extract_langchain_tool_calls(message: BaseMessage) -> list[DemoToolCall]:
    raw_calls = getattr(message, "tool_calls", [])
    if not isinstance(raw_calls, list):
        return []
    calls: list[DemoToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        name = str(raw_call.get("name", "")).strip()
        args = _parse_args(raw_call.get("args", {}))
        if name:
            calls.append(DemoToolCall(name=name, args=args, source="llm"))
    return calls


def _extract_openai_tool_calls(message: BaseMessage) -> list[DemoToolCall]:
    additional_kwargs = getattr(message, "additional_kwargs", {})
    if not isinstance(additional_kwargs, dict):
        return []
    raw_calls = additional_kwargs.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return []
    calls: list[DemoToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function", {})
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        args = _parse_args(function.get("arguments", {}))
        if name:
            calls.append(DemoToolCall(name=name, args=args, source="llm"))
    return calls


def _parse_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    return {}


def _preview_tool_result(name: str, result: Any) -> dict[str, Any]:
    if name == "summarize_candidate_data" and isinstance(result, dict):
        summary = result.get("data_summary", {})
        return {
            "case_path": result.get("case_path", ""),
            "data_summary": summary if isinstance(summary, dict) else {},
        }
    if name == "load_experiment_history" and isinstance(result, dict):
        history = result.get("local_history_summary", {})
        evidence = result.get("evidence_profile", {})
        records = result.get("experiment_records", [])
        return {
            "record_count": len(records) if isinstance(records, list) else 0,
            "local_history_summary": history if isinstance(history, dict) else {},
            "evidence_profile": evidence if isinstance(evidence, dict) else {},
        }
    if name == "get_strategy_catalog" and isinstance(result, list):
        catalog = [item for item in result if isinstance(item, dict)]
        return {
            "strategy_count": len(catalog),
            "strategies": [
                {
                    "name": item.get("name", ""),
                    "seed_config_count": len(item.get("seed_configs", []))
                    if isinstance(item.get("seed_configs", []), list)
                    else 0,
                }
                for item in catalog
            ],
        }
    if isinstance(result, dict):
        return result
    return {"result": str(result)}
