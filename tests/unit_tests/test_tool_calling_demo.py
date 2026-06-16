import json

import pytest
from langchain_core.messages import AIMessage

from autoresearch_agent.tool_calling_demo import (
    DEMO_TOOL_NAMES,
    DemoToolCall,
    demo_tool_schemas,
    execute_demo_tool_calls,
    extract_tool_calls,
    run_replay_tool_calling_demo,
)


def test_demo_tool_schemas_are_safe_subset() -> None:
    schemas = demo_tool_schemas()
    names = {schema["function"]["name"] for schema in schemas}

    assert names == set(DEMO_TOOL_NAMES)
    assert "run_strategy_evaluation" not in names
    assert all(schema["type"] == "function" for schema in schemas)


def test_replay_tool_calling_demo_executes_read_only_tools(tmp_path) -> None:
    result = run_replay_tool_calling_demo(
        data_file="data/large_seed301.txt",
        experiments_dir=str(tmp_path),
    )

    assert result.mode == "replay"
    assert [call.name for call in result.tool_calls] == list(DEMO_TOOL_NAMES)
    assert len(result.tool_results) == 3
    data_preview = result.tool_results[0].preview["data_summary"]
    assert data_preview["candidate_count"] == 33780
    history_preview = result.tool_results[1].preview
    assert history_preview["record_count"] == 0
    catalog_preview = result.tool_results[2].preview
    assert catalog_preview["strategy_count"] >= 1


def test_extract_tool_calls_from_openai_message_shape() -> None:
    message = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "summarize_candidate_data",
                        "arguments": json.dumps(
                            {"data_file": "data/large_seed301.txt"}
                        ),
                    },
                }
            ]
        },
    )

    calls = extract_tool_calls(message)

    assert calls == [
        DemoToolCall(
            name="summarize_candidate_data",
            args={"data_file": "data/large_seed301.txt"},
            source="llm",
        )
    ]


def test_execute_demo_tool_calls_rejects_non_demo_tool() -> None:
    with pytest.raises(ValueError, match="tool is not allowed"):
        execute_demo_tool_calls(
            [
                DemoToolCall(
                    name="run_strategy_evaluation",
                    args={},
                )
            ]
        )
