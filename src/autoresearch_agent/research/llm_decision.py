"""Structured LLM decision parsing and validation for the research loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

from autoresearch_agent.research.autopilot import ResearchDecision, SearchSpace
from autoresearch_agent.research.strategy_space import (
    build_strategy_configs,
    default_strategy_output_dir,
    list_strategy_space_names,
    strategy_config_from_dict,
    strategy_primitive_schema,
    strategy_space_catalog,
    validate_strategy_config,
)

MIN_CANDIDATE_LIMIT = 1
MAX_CANDIDATE_LIMIT = 8

@dataclass(frozen=True)
class LLMDecisionValidation:
    """Validated LLM decision payload and validation diagnostics."""

    decision: ResearchDecision | None
    raw_text: str
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return whether a usable LLM decision was produced."""
        return self.decision is not None and not self.errors


def validate_llm_research_decision(
    raw_text: str,
) -> LLMDecisionValidation:
    """Parse and validate one LLM-produced research decision JSON object."""
    errors: list[str] = []
    payload = _parse_json_object(raw_text, errors)
    if payload is None:
        return LLMDecisionValidation(
            decision=None,
            raw_text=raw_text,
            payload={},
            errors=errors,
        )

    search_space = str(payload.get("search_space", "")).strip()
    if not search_space:
        errors.append("search_space is required")
    available = set(list_strategy_space_names())
    if search_space and search_space not in available and search_space != "stop_search":
        errors.append(f"unknown search_space: {search_space}")

    hypothesis = str(payload.get("hypothesis", "")).strip()
    if not hypothesis:
        errors.append("hypothesis is required")

    if "param_region" in payload:
        errors.append(
            "param_region is no longer supported; use config_ids or configs"
        )
    if "selected_profiles" in payload:
        errors.append("selected_profiles is no longer supported; use config_ids")
    if "spec_ids" in payload:
        errors.append("spec_ids is no longer supported; use config_ids")
    selected_config_ids = _parse_string_list(
        payload.get("config_ids", []),
        field_name="config_ids",
        errors=errors,
    )
    inline_configs = _parse_config_list(payload.get("configs", []), errors=errors)

    candidate_limit = _parse_candidate_limit(payload.get("candidate_limit"), errors)
    expected_evidence = _parse_string_list(
        payload.get("expected_evidence"),
        field_name="expected_evidence",
        errors=errors,
    )
    reasons = _parse_string_list(
        payload.get("reasons", []),
        field_name="reasons",
        errors=errors,
    )
    failure_mode = str(payload.get("failure_mode", "")).strip()
    if not failure_mode:
        errors.append("failure_mode is required")
    strategy_family = str(payload.get("strategy_family", search_space)).strip()
    evidence_score = _parse_float(
        payload.get("evidence_score", 0.0),
        errors=errors,
        field_name="evidence_score",
    )

    recommended_solver_path = str(payload.get("recommended_solver_path", "")).strip()
    output_dir = ""
    if search_space == "stop_search":
        if not recommended_solver_path:
            errors.append("stop_search requires recommended_solver_path")
        if selected_config_ids:
            errors.append("stop_search must not include config_ids")
        if inline_configs:
            errors.append("stop_search must not include configs")
    elif search_space in available:
        output_dir = default_strategy_output_dir(search_space)
        _validate_executable_config_selection(
            search_space,
            selected_config_ids,
            inline_configs,
            errors,
        )

    if errors:
        return LLMDecisionValidation(
            decision=None,
            raw_text=raw_text,
            payload=payload,
            errors=errors,
        )

    decision = ResearchDecision(
        search_space=cast(SearchSpace, search_space),
        hypothesis=hypothesis,
        reasons=reasons or ["LLM selected this local strategy-space action."],
        expected_evidence=expected_evidence,
        output_dir=output_dir,
        candidate_limit=candidate_limit,
        recommended_solver_path=recommended_solver_path,
        failure_mode=failure_mode,
        strategy_family=strategy_family,
        param_region={},
        selected_config_ids=selected_config_ids,
        configs=inline_configs,
        evidence_score=evidence_score,
    )
    return LLMDecisionValidation(
        decision=decision,
        raw_text=raw_text,
        payload=payload,
        errors=[],
    )


def _parse_json_object(raw_text: str, errors: list[str]) -> dict[str, Any] | None:
    stripped = raw_text.strip()
    if not stripped:
        errors.append("LLM decision response is empty")
        return None
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        errors.append(f"LLM decision response is not valid JSON: {exc.msg}")
        return None
    if not isinstance(parsed, dict):
        errors.append("LLM decision response must be a JSON object")
        return None
    return cast(dict[str, Any], parsed)


def _strip_code_fence(value: str) -> str:
    lines = value.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _validate_executable_config_selection(
    search_space: str,
    selected_config_ids: list[str],
    inline_configs: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not selected_config_ids and not inline_configs:
        errors.append(
            "non-stop decisions require config_ids from seed_configs or inline configs"
        )
        return
    supported_config_ids = {
        config.config_id for config in build_strategy_configs(search_space)
    }
    unknown_config_ids = sorted(set(selected_config_ids) - supported_config_ids)
    if unknown_config_ids:
        errors.append(
            "config_ids contains unknown config_id(s) for "
            f"{search_space}: {', '.join(unknown_config_ids)}"
        )
    for config in inline_configs:
        try:
            validate_strategy_config(strategy_config_from_dict(config))
        except ValueError as exc:
            errors.append(f"invalid inline strategy config: {exc}")


def llm_decision_allowed_values(search_space: str = "") -> dict[str, Any]:
    """Return allowed catalog values for initial prompting or one repair pass."""
    catalog = strategy_space_catalog()
    available = [item["name"] for item in catalog]
    if search_space in available:
        item = next(entry for entry in catalog if entry["name"] == search_space)
        return {
            "search_spaces": [*available, "stop_search"],
            "selected_search_space": search_space,
            "config_ids": [
                seed_config["config_id"] for seed_config in item["seed_configs"]
            ],
            "key_param_values": item["key_param_values"],
            "seed_configs": item["seed_configs"],
            "primitive_schema": strategy_primitive_schema(),
        }
    return {
        "search_spaces": [*available, "stop_search"],
        "catalog": catalog,
        "primitive_schema": strategy_primitive_schema(),
    }


def _parse_candidate_limit(value: Any, errors: list[str]) -> int:
    if value is None:
        return 4
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append("candidate_limit must be an integer")
        return 4
    if value < MIN_CANDIDATE_LIMIT or value > MAX_CANDIDATE_LIMIT:
        errors.append(
            f"candidate_limit must be between {MIN_CANDIDATE_LIMIT} and {MAX_CANDIDATE_LIMIT}"
        )
        return 4
    return value


def _parse_string_list(
    value: Any,
    *,
    field_name: str,
    errors: list[str],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return []
    items = []
    for item in value:
        if not isinstance(item, str):
            errors.append(f"{field_name} values must be strings")
            continue
        stripped = item.strip()
        if stripped:
            items.append(stripped)
    return items


def _parse_config_list(value: Any, *, errors: list[str]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append("configs must be a list")
        return []
    configs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            errors.append("configs values must be objects")
            continue
        configs.append(cast(dict[str, Any], item))
    return configs


def _parse_float(value: Any, *, errors: list[str], field_name: str) -> float:
    if _is_number(value):
        return float(value)
    errors.append(f"{field_name} must be numeric")
    return 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
