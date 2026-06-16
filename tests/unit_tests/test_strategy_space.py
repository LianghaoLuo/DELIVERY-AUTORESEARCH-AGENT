from pathlib import Path

import autoresearch_agent.research.strategy_space as strategy_space_module
from autoresearch_agent.research.strategy_space import (
    ControlConfig,
    PrimaryConfig,
    RepairStep,
    StrategyConfig,
    build_strategy_configs,
    default_strategy_output_dir,
    list_strategy_space_names,
    materialize_strategy_configs,
    parse_alpha_list,
    parse_strategy_metadata_from_variant_path,
    read_strategy_config_header,
    stable_strategy_config_id,
    strategy_config_from_dict,
    strategy_config_ids_for_region,
    strategy_config_to_dict,
    strategy_primitive_schema,
    strategy_space_catalog,
    strategy_sweep_label,
)
from autoresearch_agent.solver_dev.runner import run_solver_case


def test_old_strategy_spec_api_is_removed() -> None:
    for name in (
        "StrategySpec",
        "build_strategy_specs",
        "materialize_strategy_variants",
        "render_strategy_solver",
        "stable_strategy_spec_id",
        "filter_strategy_specs_by_param_region",
        "filter_strategy_specs_by_selection",
        "filter_strategy_specs_by_config_ids",
    ):
        assert not hasattr(strategy_space_module, name)


def test_parse_and_extract_alpha_values() -> None:
    assert parse_alpha_list("0,25,100.5") == [0.0, 25.0, 100.5]


def test_strategy_config_canonicalizes_and_hashes_primitives() -> None:
    config = StrategyConfig(
        name="unit primitive",
        family="unit",
        intent="unit_local_improve",
        source="llm",
        primary=PrimaryConfig(kind="willingness_adjusted", params={"alpha": 90.0}),
        repairs=(RepairStep(kind="local_improve", params={"max_passes": 1}),),
        control=ControlConfig(time_budget_seconds=8.5),
        tags=("smoke",),
    )
    payload = strategy_config_to_dict(config)
    parsed = strategy_config_from_dict(payload)

    assert parsed == config
    assert stable_strategy_config_id(parsed) == stable_strategy_config_id(config)
    assert stable_strategy_config_id(config).startswith("cfg_")
    assert "legacy_metadata" not in payload


def test_strategy_config_rejects_invalid_primitives_params_and_controls() -> None:
    try:
        PrimaryConfig(kind="missing")
    except ValueError as exc:
        assert "unknown primary primitive" in str(exc)
    else:
        raise AssertionError("invalid primary primitive was accepted")

    try:
        RepairStep(kind="missing")
    except ValueError as exc:
        assert "unknown repair primitive" in str(exc)
    else:
        raise AssertionError("invalid repair primitive was accepted")

    try:
        PrimaryConfig(kind="greedy", params={"alpha": 90.0})
    except ValueError as exc:
        assert "unknown greedy param" in str(exc)
    else:
        raise AssertionError("invalid primitive param was accepted")

    try:
        ControlConfig(max_extra_dispatches=-1)
    except ValueError as exc:
        assert "max_extra_dispatches" in str(exc)
    else:
        raise AssertionError("invalid control bounds were accepted")


def test_strategy_space_registry_and_catalog_are_config_first() -> None:
    names = list_strategy_space_names()
    catalog = strategy_space_catalog()

    assert names == [
        "broad_strategy",
        "local_improve",
        "duplicate_augment",
        "duplicate_augment_refine",
        "risk_tier_duplicate",
        "low_willingness_deep_duplicate",
        "task_risk_duplicate",
        "bundle_split_duplicate",
        "bundle_merge_duplicate",
        "pressure_targeted",
        "portfolio_overlay",
        "beam_staged",
    ]
    assert [item["name"] for item in catalog] == names
    for name in names:
        configs = build_strategy_configs(name, alphas=[90.0])
        catalog_item = next(item for item in catalog if item["name"] == name)

        assert configs
        assert strategy_sweep_label(name).endswith("strategy-sweep")
        assert default_strategy_output_dir(name).endswith(name)
        assert catalog_item["seed_config_count"] == len(build_strategy_configs(name))
        assert catalog_item["seed_configs"]
        assert all(item["config_id"] for item in catalog_item["seed_configs"])
        assert all("pipeline" in item for item in catalog_item["seed_configs"])
        assert "primitive_schema" not in catalog_item
        assert "legacy_strategy_type" not in str(catalog_item)

    schema = strategy_primitive_schema()
    assert "regret_rank" in schema["primary_primitives"]
    assert "staged_duplicate" in schema["repair_primitives"]


def test_build_strategy_configs_filters_by_config_id_and_inline_configs() -> None:
    profile_configs = [
        config
        for config in build_strategy_configs("bundle_merge_duplicate")
        if "explore_alpha92p5_m010" in config.name
    ]
    inline_config = StrategyConfig(
        name="inline_risk",
        family="risk_tier_duplicate",
        intent="inline_risk_probe",
        source="llm",
        primary=PrimaryConfig(kind="willingness_adjusted", params={"alpha": 90.0}),
        repairs=(RepairStep(kind="risk_tier_duplicate"),),
        control=ControlConfig(max_extra_dispatches=10, max_couriers_per_assignment=2),
    )
    selected = build_strategy_configs(
        "risk_tier_duplicate",
        inline_configs=[strategy_config_to_dict(inline_config)],
        selected_config_ids=[inline_config.config_id],
    )

    assert profile_configs
    assert build_strategy_configs(
        "bundle_merge_duplicate",
        selected_config_ids=[profile_configs[0].config_id],
    ) == [profile_configs[0]]
    assert selected == [inline_config]


def test_strategy_config_ids_for_region_maps_evidence_to_configs() -> None:
    config_ids = strategy_config_ids_for_region(
        "bundle_merge_duplicate",
        {
            "profile": "bundle_merge_focused",
            "merge_min_improvement_values": [0.0, -10.0],
            "max_extra_dispatches_min": 20,
            "max_extra_dispatches_max": 30,
            "max_couriers_per_assignment_max": 3,
        },
    )

    assert config_ids
    configs = build_strategy_configs(
        "bundle_merge_duplicate",
        selected_config_ids=config_ids,
    )
    assert all(config.family == "bundle_merge_duplicate" for config in configs)


def test_materialized_configs_have_headers_and_metadata(tmp_path: Path) -> None:
    config = build_strategy_configs("local_improve")[0]
    path = materialize_strategy_configs([config], tmp_path)[0]

    assert read_strategy_config_header(path) == config
    metadata = parse_strategy_metadata_from_variant_path(path)
    assert metadata["config_id"] == config.config_id
    assert metadata["family"] == config.family
    assert metadata["pipeline"] == config.pipeline
    assert "def solve(input_text):" in path.read_text()


def test_each_search_space_materializes_a_valid_solver(tmp_path: Path) -> None:
    for name in list_strategy_space_names():
        config = build_strategy_configs(name, alphas=[90.0])[0]
        path = materialize_strategy_configs([config], tmp_path / name)[0]
        source = path.read_text()
        assert "StrategyConfig:" in source
        assert "autoresearch_agent" not in source
        assert "langchain" not in source.lower()
        assert "langgraph" not in source.lower()
        result = run_solver_case(
            str(path),
            "data/large_seed301.txt",
            timeout_seconds=10.0,
        )
        assert result.validation.is_valid, result.validation.errors
        assert not result.timed_out


def test_inline_primitive_combinations_are_renderable_and_valid(tmp_path: Path) -> None:
    configs = [
        StrategyConfig(
            name="inline_local",
            family="inline",
            intent="local_repair",
            primary=PrimaryConfig("willingness_adjusted", {"alpha": 90.0}),
            repairs=(RepairStep("local_improve", {"max_passes": 1}),),
            control=ControlConfig(time_budget_seconds=8.5),
        ),
        StrategyConfig(
            name="inline_bundle_risk",
            family="inline",
            intent="bundle_risk",
            primary=PrimaryConfig(
                "bundle_merge",
                {"alpha": 92.5, "scarce_bundle_bonus": 50.0},
            ),
            repairs=(
                RepairStep("risk_tier_duplicate", {"max_extra_dispatches": 10}),
            ),
            control=ControlConfig(max_extra_dispatches=10, max_couriers_per_assignment=2),
        ),
        StrategyConfig(
            name="inline_beam_stage",
            family="inline",
            intent="beam_stage",
            primary=PrimaryConfig("beam", {"beam_width": 8}),
            repairs=(
                RepairStep("risk_tier_duplicate", {"max_extra_dispatches": 10}),
                RepairStep("staged_duplicate", {"staged_max_extra_dispatches": 5}),
            ),
            control=ControlConfig(max_extra_dispatches=10, max_couriers_per_assignment=2),
        ),
        StrategyConfig(
            name="inline_regret",
            family="inline",
            intent="regret_risk",
            primary=PrimaryConfig(
                "regret_rank",
                {"regret_score_weight": 0.2, "regret_scarcity_weight": 90.0},
            ),
            repairs=(RepairStep("risk_tier_duplicate", {"max_extra_dispatches": 10}),),
            control=ControlConfig(max_extra_dispatches=10, max_couriers_per_assignment=2),
        ),
        StrategyConfig(
            name="inline_overlay",
            family="inline",
            intent="task_overlay",
            primary=PrimaryConfig("willingness_adjusted", {"alpha": 90.0}),
            repairs=(RepairStep("task_overlay", {"max_extra_dispatches": 10}),),
            control=ControlConfig(max_extra_dispatches=10, max_couriers_per_assignment=2),
        ),
    ]

    paths = materialize_strategy_configs(configs, tmp_path)

    assert len(paths) == len(configs)
    for path in paths:
        result = run_solver_case(
            str(path),
            "data/large_seed301.txt",
            timeout_seconds=10.0,
        )
        assert result.validation.is_valid, result.validation.errors
        assert not result.timed_out
