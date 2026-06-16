import json

from autoresearch_agent.research.llm_decision import validate_llm_research_decision
from autoresearch_agent.research.strategy_space import build_strategy_configs


def test_validate_llm_research_decision_accepts_executable_catalog_choice() -> None:
    config_id = _seed_config_id("risk_tier_duplicate", "low_willingness_aggressive")
    payload = {
        "search_space": "risk_tier_duplicate",
        "hypothesis": "Probe low-willingness duplicate dispatch.",
        "config_ids": [config_id],
        "configs": [],
        "candidate_limit": 2,
        "expected_evidence": ["robust-suite mean proxy"],
        "failure_mode": "low_expected_success",
        "strategy_family": "risk_tier_duplicate",
        "reasons": ["local duplicate augmentation has diminishing returns"],
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert result.is_valid
    assert result.decision is not None
    assert result.decision.search_space == "risk_tier_duplicate"
    assert result.decision.candidate_limit == 2
    assert result.decision.output_dir.endswith("/risk_tier_duplicate")
    assert result.decision.selected_config_ids == [config_id]
    assert result.errors == []


def test_validate_llm_research_decision_accepts_inline_config() -> None:
    payload = {
        "search_space": "local_improve",
        "hypothesis": "Try a simple primitive config.",
        "config_ids": [],
        "configs": [
            {
                "name": "inline_local_improve",
                "family": "local_improve",
                "intent": "inline_local_repair",
                "source": "llm",
                "primary": {"kind": "willingness_adjusted", "params": {"alpha": 90.0}},
                "repairs": [{"kind": "local_improve", "params": {"max_passes": 1}}],
                "control": {"time_budget_seconds": 8.5},
                "tags": ["inline"],
            }
        ],
        "candidate_limit": 2,
        "expected_evidence": ["robust-suite mean proxy"],
        "failure_mode": "high_score_cost",
        "strategy_family": "local_improve",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert result.is_valid
    assert result.decision is not None
    assert result.decision.configs
    assert result.decision.selected_config_ids == []


def test_validate_llm_research_decision_rejects_invalid_json() -> None:
    result = validate_llm_research_decision("not json")

    assert not result.is_valid
    assert result.decision is None
    assert "not valid JSON" in result.errors[0]


def test_validate_llm_research_decision_rejects_unknown_search_space() -> None:
    payload = {
        "search_space": "invent_new_solver",
        "hypothesis": "Try an unknown action.",
        "selected_profiles": [],
        "spec_ids": [],
        "candidate_limit": 2,
        "expected_evidence": [],
        "failure_mode": "unknown",
        "strategy_family": "unknown",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any("unknown search_space" in error for error in result.errors)


def test_validate_llm_research_decision_rejects_legacy_profile_selector() -> None:
    payload = {
        "search_space": "risk_tier_duplicate",
        "hypothesis": "Pick an old profile selector.",
        "selected_profiles": ["low_willingness_aggressive"],
        "config_ids": [],
        "candidate_limit": 2,
        "expected_evidence": [],
        "failure_mode": "low_expected_success",
        "strategy_family": "risk_tier_duplicate",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any("selected_profiles is no longer supported" in error for error in result.errors)


def test_validate_llm_research_decision_rejects_unknown_config_id() -> None:
    payload = {
        "search_space": "risk_tier_duplicate",
        "hypothesis": "Pick an unavailable config.",
        "config_ids": ["cfg_missing"],
        "configs": [],
        "candidate_limit": 2,
        "expected_evidence": [],
        "failure_mode": "low_expected_success",
        "strategy_family": "risk_tier_duplicate",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any("unknown config_id" in error for error in result.errors)


def test_validate_llm_research_decision_rejects_param_region() -> None:
    payload = {
        "search_space": "risk_tier_duplicate",
        "hypothesis": "Use the old free-form selector.",
        "param_region": {"profile": "low_willingness_aggressive"},
        "config_ids": [_seed_config_id("risk_tier_duplicate", "low_willingness_aggressive")],
        "configs": [],
        "candidate_limit": 2,
        "expected_evidence": [],
        "failure_mode": "low_expected_success",
        "strategy_family": "risk_tier_duplicate",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any(
        "param_region is no longer supported" in error for error in result.errors
    )


def test_validate_llm_research_decision_rejects_empty_selector() -> None:
    payload = {
        "search_space": "risk_tier_duplicate",
        "hypothesis": "Do not select anything concrete.",
        "config_ids": [],
        "configs": [],
        "candidate_limit": 2,
        "expected_evidence": [],
        "failure_mode": "low_expected_success",
        "strategy_family": "risk_tier_duplicate",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any(
        "require config_ids" in error for error in result.errors
    )


def test_validate_llm_research_decision_rejects_missing_failure_mode() -> None:
    payload = {
        "search_space": "risk_tier_duplicate",
        "hypothesis": "Omit decision evidence fields.",
        "config_ids": [_seed_config_id("risk_tier_duplicate", "low_willingness_aggressive")],
        "configs": [],
        "candidate_limit": 2,
        "expected_evidence": [],
        "strategy_family": "risk_tier_duplicate",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any("failure_mode is required" in error for error in result.errors)


def test_validate_llm_research_decision_requires_explicit_stop_solver_path() -> None:
    payload = {
        "search_space": "stop_search",
        "hypothesis": "Stop without naming the solver.",
        "config_ids": [],
        "configs": [],
        "candidate_limit": 2,
        "expected_evidence": ["stop evidence"],
        "failure_mode": "stagnation",
        "strategy_family": "stop_search",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any(
        "stop_search requires recommended_solver_path" in error
        for error in result.errors
    )


def test_validate_llm_research_decision_rejects_candidate_limit_out_of_range() -> None:
    payload = {
        "search_space": "risk_tier_duplicate",
        "hypothesis": "Ask for too many candidates.",
        "config_ids": [_seed_config_id("risk_tier_duplicate", "low_willingness_aggressive")],
        "configs": [],
        "candidate_limit": 99,
        "expected_evidence": [],
        "failure_mode": "low_expected_success",
        "strategy_family": "risk_tier_duplicate",
    }

    result = validate_llm_research_decision(json.dumps(payload))

    assert not result.is_valid
    assert result.decision is None
    assert any("candidate_limit must be between" in error for error in result.errors)


def _seed_config_id(search_space: str, profile: str) -> str:
    for config in build_strategy_configs(search_space):
        if profile in config.name or profile in config.tags:
            return config.config_id
    raise AssertionError(f"missing seed profile {profile}")
