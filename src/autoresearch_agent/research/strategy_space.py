"""Config-first solver strategy generation for local AutoResearch sweeps."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from autoresearch_agent.research.strategy_renderer import render_solver_source
from autoresearch_agent.solver_dev.variants import VariantSuiteBatchResult

SOLVER_SCORE_WEIGHT = 1.0
SOLVER_EXPECTED_FAILED_WEIGHT = 0.0
SOLVER_EXPECTED_SUCCESS_CREDIT = 100.0

PRIMARY_PRIMITIVES = {
    "greedy",
    "willingness_adjusted",
    "bundle_merge",
    "bundle_split",
    "beam",
    "regret_rank",
}
REPAIR_PRIMITIVES = {
    "local_improve",
    "duplicate_dispatch",
    "risk_tier_duplicate",
    "task_overlay",
    "staged_duplicate",
    "drop_tail_probe",
}
PRIMARY_PARAM_KEYS = {
    "greedy": {"score_weight"},
    "willingness_adjusted": {
        "alpha",
        "score_weight",
        "failure_weight",
        "success_credit",
        "expected_failed_weight",
        "expected_success_credit",
        "bundle_bias",
        "low_willingness_alpha",
    },
    "bundle_merge": {
        "alpha",
        "low_willingness_alpha",
        "score_weight",
        "expected_failed_weight",
        "expected_success_credit",
        "bundle_bias",
        "scarce_alpha",
        "scarce_bundle_bonus",
        "scarce_ratio_threshold",
        "scarce_use_merge",
        "merge_min_improvement",
    },
    "bundle_split": {
        "alpha",
        "score_weight",
        "expected_failed_weight",
        "expected_success_credit",
        "split_min_improvement",
        "bundle_bias",
    },
    "beam": {
        "alpha",
        "low_willingness_alpha",
        "beam_width",
        "beam_top_per_task",
        "beam_global_limit",
        "beam_success_credit",
        "beam_pair_penalty",
        "beam_bundle_penalty",
        "beam_mode",
        "beam_scope",
        "scarce_alpha",
        "scarce_bundle_bonus",
        "scarce_ratio_threshold",
    },
    "regret_rank": {
        "alpha",
        "low_willingness_alpha",
        "regret_score_weight",
        "regret_scarcity_weight",
        "regret_willingness_weight",
        "regret_bundle_bonus",
    },
}
REPAIR_PARAM_KEYS = {
    "local_improve": {
        "alpha",
        "max_passes",
        "time_budget_seconds",
        "score_weight",
        "expected_failed_weight",
        "expected_success_credit",
    },
    "duplicate_dispatch": {
        "min_success_probability",
        "min_roi",
        "max_extra_dispatches",
        "max_couriers_per_assignment",
        "score_weight",
        "expected_failed_weight",
        "expected_success_credit",
        "time_budget_seconds",
    },
    "risk_tier_duplicate": {
        "alpha",
        "low_willingness_alpha",
        "high_risk_target",
        "mid_risk_target",
        "min_roi",
        "high_success_min_roi",
        "max_extra_dispatches",
        "max_couriers_per_assignment",
        "score_weight",
        "expected_failed_weight",
        "expected_success_credit",
        "time_budget_seconds",
        "low_willingness_success_threshold",
        "low_willingness_assignment_ratio",
        "scarce_max_extra_dispatches",
        "budget_high_risk_target",
        "budget_mid_risk_target",
        "budget_min_roi",
        "budget_max_extra_dispatches",
        "budget_max_couriers_per_assignment",
    },
    "task_overlay": {
        "high_risk_target",
        "mid_risk_target",
        "min_roi",
        "max_extra_dispatches",
        "max_couriers_per_assignment",
        "task_overlay_max_extra_dispatches",
        "task_overlay_max_couriers",
        "task_overlay_min_roi",
        "scarce_task_overlay_extra",
        "portfolio_success_credit",
        "portfolio_pair_penalty",
        "portfolio_bundle_penalty",
        "high_success_min_roi",
        "score_weight",
        "expected_success_credit",
        "expected_failed_weight",
    },
    "staged_duplicate": {
        "staged_max_extra_dispatches",
        "staged_max_couriers",
        "staged_tail_fraction",
        "staged_tail_success_threshold",
        "staged_target",
        "staged_min_roi",
        "scarce_staged_extra",
        "max_extra_dispatches",
        "max_couriers_per_assignment",
        "min_roi",
        "high_risk_target",
        "mid_risk_target",
    },
    "drop_tail_probe": {"drop_ratio", "max_drop_tasks", "min_task_count"},
}
CONTROL_PARAM_KEYS = {
    "max_extra_dispatches",
    "max_couriers_per_assignment",
    "min_roi",
    "high_risk_target",
    "mid_risk_target",
    "time_budget_seconds",
    "max_merge_passes",
    "max_split_passes",
    "merge_min_improvement",
    "split_min_improvement",
    "scarce_max_extra_dispatches",
    "low_willingness_success_threshold",
    "low_willingness_assignment_ratio",
    "beam_width",
    "beam_top_per_task",
    "beam_global_limit",
    "beam_success_credit",
    "beam_pair_penalty",
    "beam_bundle_penalty",
    "beam_mode",
    "beam_scope",
    "staged_max_extra_dispatches",
    "staged_max_couriers",
    "staged_tail_fraction",
    "staged_tail_success_threshold",
    "staged_target",
    "staged_min_roi",
    "scarce_staged_extra",
}
CATALOG_PARAM_KEYS = sorted(
    set().union(*PRIMARY_PARAM_KEYS.values(), *REPAIR_PARAM_KEYS.values())
    | CONTROL_PARAM_KEYS
)
CONFIG_HEADER_PREFIX = "# StrategyConfig: "


@dataclass(frozen=True)
class PrimaryConfig:
    """Primary assignment primitive configuration."""

    kind: str
    params: dict[str, float | int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the primary primitive."""
        kind = self.kind.strip()
        if kind not in PRIMARY_PRIMITIVES:
            raise ValueError(f"unknown primary primitive: {kind}")
        _validate_param_keys(kind, self.params, PRIMARY_PARAM_KEYS)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "params", _canonical_scalar_dict(self.params))


@dataclass(frozen=True)
class RepairStep:
    """One ordered repair primitive in the generated solver pipeline."""

    kind: str
    params: dict[str, float | int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the repair primitive."""
        kind = self.kind.strip()
        if kind not in REPAIR_PRIMITIVES:
            raise ValueError(f"unknown repair primitive: {kind}")
        _validate_param_keys(kind, self.params, REPAIR_PARAM_KEYS)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "params", _canonical_scalar_dict(self.params))


@dataclass(frozen=True)
class ControlConfig:
    """Shared constraints and numeric controls for a strategy config."""

    max_extra_dispatches: int | None = None
    max_couriers_per_assignment: int | None = None
    min_roi: float | None = None
    high_risk_target: float | None = None
    mid_risk_target: float | None = None
    time_budget_seconds: float | None = None
    params: dict[str, float | int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate common control bounds."""
        if self.max_extra_dispatches is not None and self.max_extra_dispatches < 0:
            raise ValueError("max_extra_dispatches must be non-negative")
        if (
            self.max_couriers_per_assignment is not None
            and self.max_couriers_per_assignment < 1
        ):
            raise ValueError("max_couriers_per_assignment must be at least 1")
        if self.high_risk_target is not None:
            _validate_probability("high_risk_target", self.high_risk_target)
        if self.mid_risk_target is not None:
            _validate_probability("mid_risk_target", self.mid_risk_target)
        if self.time_budget_seconds is not None and self.time_budget_seconds <= 0.0:
            raise ValueError("time_budget_seconds must be positive")
        unknown = set(self.params) - CONTROL_PARAM_KEYS
        if unknown:
            raise ValueError(f"unknown control param(s): {', '.join(sorted(unknown))}")
        object.__setattr__(self, "params", _canonical_scalar_dict(self.params))


@dataclass(frozen=True)
class StrategyConfig:
    """Composable solver strategy configuration used by AutoResearch."""

    name: str
    family: str
    intent: str
    primary: PrimaryConfig
    source: str = "seed_config"
    repairs: tuple[RepairStep, ...] = ()
    control: ControlConfig = field(default_factory=ControlConfig)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize nested config values and validate basic shape."""
        name = self.name.strip()
        family = self.family.strip()
        intent = self.intent.strip()
        source = self.source.strip() or "seed_config"
        if not name:
            raise ValueError("strategy config name is required")
        if not family:
            raise ValueError("strategy config family is required")
        if not intent:
            raise ValueError("strategy config intent is required")
        tags = tuple(sorted({str(tag).strip() for tag in self.tags if str(tag).strip()}))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "repairs", tuple(self.repairs))
        object.__setattr__(self, "tags", tags)

    @property
    def config_id(self) -> str:
        """Return the stable canonical ID for this config."""
        return stable_strategy_config_id(self)

    @property
    def pipeline(self) -> str:
        """Return a compact primitive pipeline label."""
        steps = [self.primary.kind, *[repair.kind for repair in self.repairs]]
        return " -> ".join(steps)


@dataclass(frozen=True)
class StrategySpaceDefinition:
    """One registered local solver search space."""

    name: str
    label: str
    output_dir: str
    notes: str
    build_configs: Callable[[], list[StrategyConfig]]
    description: str = ""
    targets: tuple[str, ...] = ()
    risk_level: str = "medium"
    param_hints: tuple[str, ...] = ()


def strategy_config_to_dict(config: StrategyConfig) -> dict[str, Any]:
    """Return one strategy config as canonical JSON-ish primitives."""
    return {
        "name": config.name,
        "family": config.family,
        "intent": config.intent,
        "source": config.source,
        "primary": {
            "kind": config.primary.kind,
            "params": _canonical_jsonish_dict(config.primary.params),
        },
        "repairs": [
            {
                "kind": repair.kind,
                "params": _canonical_jsonish_dict(repair.params),
            }
            for repair in config.repairs
        ],
        "control": {
            "max_extra_dispatches": config.control.max_extra_dispatches,
            "max_couriers_per_assignment": (
                config.control.max_couriers_per_assignment
            ),
            "min_roi": config.control.min_roi,
            "high_risk_target": config.control.high_risk_target,
            "mid_risk_target": config.control.mid_risk_target,
            "time_budget_seconds": config.control.time_budget_seconds,
            "params": _canonical_jsonish_dict(config.control.params),
        },
        "tags": list(config.tags),
    }


def strategy_config_from_dict(payload: dict[str, Any]) -> StrategyConfig:
    """Parse and validate a strategy config from JSON-ish primitives."""
    if not isinstance(payload, dict):
        raise ValueError("strategy config must be an object")
    primary_payload = payload.get("primary", {})
    if not isinstance(primary_payload, dict):
        raise ValueError("strategy config primary must be an object")
    control_payload = payload.get("control", {})
    if not isinstance(control_payload, dict):
        raise ValueError("strategy config control must be an object")
    repairs_payload = payload.get("repairs", [])
    if not isinstance(repairs_payload, list):
        raise ValueError("strategy config repairs must be a list")
    repairs = []
    for repair_payload in repairs_payload:
        if not isinstance(repair_payload, dict):
            raise ValueError("repair step must be an object")
        repairs.append(
            RepairStep(
                kind=str(repair_payload.get("kind", "")).strip(),
                params=_parse_scalar_dict(repair_payload.get("params", {})),
            )
        )
    tags_payload = payload.get("tags", [])
    if not isinstance(tags_payload, list):
        raise ValueError("strategy config tags must be a list")
    return StrategyConfig(
        name=str(payload.get("name", "")).strip(),
        family=str(payload.get("family", "")).strip(),
        intent=str(payload.get("intent", "")).strip(),
        source=str(payload.get("source", "llm")).strip(),
        primary=PrimaryConfig(
            kind=str(primary_payload.get("kind", "")).strip(),
            params=_parse_scalar_dict(primary_payload.get("params", {})),
        ),
        repairs=tuple(repairs),
        control=ControlConfig(
            max_extra_dispatches=_optional_int(
                control_payload.get("max_extra_dispatches")
            ),
            max_couriers_per_assignment=_optional_int(
                control_payload.get("max_couriers_per_assignment")
            ),
            min_roi=_optional_float(control_payload.get("min_roi")),
            high_risk_target=_optional_float(
                control_payload.get("high_risk_target")
            ),
            mid_risk_target=_optional_float(control_payload.get("mid_risk_target")),
            time_budget_seconds=_optional_float(
                control_payload.get("time_budget_seconds")
            ),
            params=_parse_scalar_dict(control_payload.get("params", {})),
        ),
        tags=tuple(str(tag).strip() for tag in tags_payload),
    )


def stable_strategy_config_id(config: StrategyConfig) -> str:
    """Return a stable ID based on the canonical strategy config JSON."""
    payload = strategy_config_to_dict(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"cfg_{hashlib.sha256(encoded).hexdigest()[:16]}"


def strategy_primitive_schema() -> dict[str, Any]:
    """Return the LLM-facing primitive schema and allowed parameter names."""
    return {
        "strategy_config": {
            "required": ["name", "family", "intent", "primary"],
            "source": "llm|seed_config",
            "tags": "list of short semantic labels",
            "primary": {
                kind: sorted(params)
                for kind, params in sorted(PRIMARY_PARAM_KEYS.items())
            },
            "repairs": {
                kind: sorted(params)
                for kind, params in sorted(REPAIR_PARAM_KEYS.items())
            },
            "control": {
                "max_extra_dispatches": "non-negative integer or null",
                "max_couriers_per_assignment": "integer >= 1 or null",
                "min_roi": "number or null",
                "high_risk_target": "probability in [0, 1] or null",
                "mid_risk_target": "probability in [0, 1] or null",
                "time_budget_seconds": "positive number or null",
                "params": sorted(CONTROL_PARAM_KEYS),
            },
        },
        "primary_primitives": sorted(PRIMARY_PRIMITIVES),
        "repair_primitives": sorted(REPAIR_PRIMITIVES),
    }


def parse_alpha_list(value: str) -> list[float]:
    """Parse a comma-separated alpha list."""
    alphas = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item:
            alphas.append(float(item))
    if not alphas:
        raise ValueError("at least one alpha value is required")
    return alphas


def build_broad_strategy_configs(
    *,
    alphas: Sequence[float] | None = None,
) -> list[StrategyConfig]:
    """Build the broad multi-family lightweight strategy config space."""
    alpha_values = list(alphas) if alphas is not None else [
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
        92.5,
        95.0,
        100.0,
        110.0,
        125.0,
    ]
    configs = [
        _base_config(
            name=f"score_alpha_{_format_param_token('a', alpha)}",
            family="broad_strategy",
            intent="bootstrap_probability_aware_greedy",
            primary_kind="greedy" if alpha == 0.0 else "willingness_adjusted",
            primary_params={"alpha": float(alpha)},
            tags=("greedy_sort",),
        )
        for alpha in alpha_values
    ]
    for failure_penalty in (100.0, 250.0, 500.0, 1000.0):
        configs.append(
            _base_config(
                name=f"failure_penalty_{_format_param_token('f', failure_penalty)}",
                family="broad_strategy",
                intent="bootstrap_failure_penalty",
                primary_kind="willingness_adjusted",
                primary_params={
                    "failure_weight": float(failure_penalty),
                    "success_credit": 0.0,
                },
                tags=("greedy_sort", "failure_penalty"),
            )
        )
    for alpha in (75.0, 90.0, 100.0):
        for bundle_bias in (-50.0, -25.0, 25.0):
            configs.append(
                _base_config(
                    name=(
                        "bundle_bias_"
                        f"{_format_param_token('a', alpha)}_"
                        f"{_format_signed_param_token('b', bundle_bias)}"
                    ),
                    family="broad_strategy",
                    intent="bootstrap_bundle_bias",
                    primary_kind="willingness_adjusted",
                    primary_params={
                        "alpha": float(alpha),
                        "bundle_bias": float(bundle_bias),
                    },
                    tags=("bundle_bias",),
                )
            )
    for score_weight, failure_weight, success_credit, bundle_bias in (
        (5.0, 250.0, 250.0, 0.0),
        (5.0, 250.0, 250.0, -25.0),
        (5.0, 500.0, 0.0, 0.0),
        (10.0, 500.0, 500.0, -25.0),
    ):
        configs.append(
            _base_config(
                name=(
                    "mixed_"
                    f"{_format_param_token('sw', score_weight)}_"
                    f"{_format_param_token('fw', failure_weight)}_"
                    f"{_format_param_token('sc', success_credit)}_"
                    f"{_format_signed_param_token('b', bundle_bias)}"
                ),
                family="broad_strategy",
                intent="bootstrap_mixed_weight",
                primary_kind="willingness_adjusted",
                primary_params={
                    "score_weight": float(score_weight),
                    "failure_weight": float(failure_weight),
                    "success_credit": float(success_credit),
                    "bundle_bias": float(bundle_bias),
                },
                tags=("parameter_search",),
            )
        )
    for alpha in (85.0, 90.0, 92.5, 100.0):
        configs.append(
            _base_config(
                name=f"local_improve_{_format_param_token('a', alpha)}",
                family="broad_strategy",
                intent="bootstrap_local_improve",
                primary_kind="willingness_adjusted",
                primary_params={"alpha": float(alpha)},
                repairs=(("local_improve", {"max_passes": 1}),),
                tags=("local_improve",),
            )
        )
    return configs


def build_local_improve_configs() -> list[StrategyConfig]:
    """Build a compact local-improvement neighborhood."""
    configs = []
    for alpha in (80.0, 85.0, 88.0, 90.0, 92.5, 95.0, 100.0):
        configs.append(_local_improve_config(alpha=alpha, max_passes=2, time_budget=8.5))
    for max_passes in (1, 3):
        configs.append(
            _local_improve_config(alpha=90.0, max_passes=max_passes, time_budget=8.5)
        )
    for time_budget in (6.5, 9.2):
        configs.append(
            _local_improve_config(alpha=90.0, max_passes=2, time_budget=time_budget)
        )
    for alpha in (85.0, 92.5):
        configs.append(_local_improve_config(alpha=alpha, max_passes=3, time_budget=9.2))
    return configs


def build_duplicate_augment_configs() -> list[StrategyConfig]:
    """Build a compact duplicate-dispatch augmentation strategy space."""
    configs = []
    for alpha in (85.0, 90.0, 95.0):
        for min_success in (0.92, 0.95):
            configs.append(
                _duplicate_config(
                    family="duplicate_augment",
                    name_prefix="duplicate_augment",
                    alpha=alpha,
                    min_success=min_success,
                    min_roi=0.0,
                    max_extra=20,
                    max_couriers=2,
                )
            )
    for max_extra in (10, 30):
        configs.append(
            _duplicate_config(
                family="duplicate_augment",
                name_prefix="duplicate_augment",
                alpha=90.0,
                min_success=0.94,
                min_roi=0.0,
                max_extra=max_extra,
                max_couriers=2,
            )
        )
    for min_roi in (-25.0, 25.0):
        configs.append(
            _duplicate_config(
                family="duplicate_augment",
                name_prefix="duplicate_augment",
                alpha=90.0,
                min_success=0.94,
                min_roi=min_roi,
                max_extra=20,
                max_couriers=2,
            )
        )
    configs.append(
        _duplicate_config(
            family="duplicate_augment",
            name_prefix="duplicate_augment",
            alpha=90.0,
            min_success=0.97,
            min_roi=-25.0,
            max_extra=30,
            max_couriers=3,
        )
    )
    return configs


def build_duplicate_augment_refine_configs() -> list[StrategyConfig]:
    """Build a duplicate-dispatch refinement strategy space."""
    configs = []
    for alpha in (85.0, 90.0, 95.0):
        for min_success in (0.88, 0.90, 0.92, 0.94, 0.96):
            configs.append(
                _duplicate_config(
                    family="duplicate_augment_refine",
                    name_prefix="duplicate_refine",
                    alpha=alpha,
                    min_success=min_success,
                    min_roi=0.0,
                    max_extra=20,
                    max_couriers=2,
                )
            )
    for min_roi in (-50.0, -25.0, 10.0, 25.0, 50.0):
        configs.append(
            _duplicate_config(
                family="duplicate_augment_refine",
                name_prefix="duplicate_refine",
                alpha=90.0,
                min_success=0.92,
                min_roi=min_roi,
                max_extra=20,
                max_couriers=2,
            )
        )
    for max_extra in (10, 15, 25, 30, 40):
        configs.append(
            _duplicate_config(
                family="duplicate_augment_refine",
                name_prefix="duplicate_refine",
                alpha=90.0,
                min_success=0.92,
                min_roi=0.0,
                max_extra=max_extra,
                max_couriers=2,
            )
        )
    return configs


def build_risk_tier_duplicate_configs() -> list[StrategyConfig]:
    """Build risk-tier duplicate-dispatch configs for low-willingness cases."""
    return [
        _risk_config("risk_tier_duplicate", "conservative", 95.0, 0.90, 0.88, 15, 2),
        _risk_config("risk_tier_duplicate", "current_plus", 95.0, 0.92, 0.90, 20, 2),
        _risk_config(
            "risk_tier_duplicate",
            "low_willingness_aggressive",
            90.0,
            0.92,
            0.90,
            30,
            3,
            low_willingness_alpha=90.0,
        ),
        _risk_config(
            "risk_tier_duplicate",
            "alpha92p5_aggressive",
            92.5,
            0.92,
            0.90,
            30,
            3,
        ),
        _risk_config(
            "risk_tier_duplicate",
            "dynamic_alpha_aggressive",
            95.0,
            0.92,
            0.90,
            30,
            3,
            low_willingness_alpha=90.0,
        ),
        _risk_config("risk_tier_duplicate", "scarce_safe", 95.0, 0.88, 0.88, 10, 2),
    ]


def build_low_willingness_deep_duplicate_configs() -> list[StrategyConfig]:
    """Build deeper risk-tier probes for low-willingness reference cases."""
    return [
        _risk_config(
            "low_willingness_deep_duplicate",
            "near_rm010_hs015",
            90.0,
            0.95,
            0.92,
            40,
            3,
            min_roi=-10.0,
            high_success_min_roi=15.0,
        ),
        _risk_config(
            "low_willingness_deep_duplicate",
            "near_rm040_hs015",
            90.0,
            0.95,
            0.92,
            50,
            3,
            min_roi=-40.0,
            high_success_min_roi=15.0,
        ),
        _risk_config(
            "low_willingness_deep_duplicate",
            "ultra_c4_rm025",
            90.0,
            0.97,
            0.93,
            60,
            4,
            min_roi=-25.0,
            high_success_min_roi=15.0,
            low_willingness_alpha=88.0,
        ),
        _risk_config(
            "low_willingness_deep_duplicate",
            "ultra_c4_rm045",
            92.5,
            0.98,
            0.95,
            60,
            4,
            min_roi=-45.0,
            high_success_min_roi=10.0,
            low_willingness_alpha=90.0,
        ),
    ]


def build_task_risk_duplicate_configs() -> list[StrategyConfig]:
    """Build task-level duplicate overlay configs."""
    return [
        _task_overlay_config("balanced", 92.5, 0.92, 0.90, 30, 3, 0.0),
        _task_overlay_config("low_willingness_aggressive", 90.0, 0.95, 0.92, 45, 3, -25.0),
        _task_overlay_config("scarce_bundle_overlay", 92.5, 0.94, 0.90, 35, 3, -10.0, bundle_bias=-10.0),
        _task_overlay_config("safe_overlay", 95.0, 0.90, 0.88, 20, 2, 5.0),
    ]


def build_bundle_split_duplicate_configs() -> list[StrategyConfig]:
    """Build bundle-split primary-search duplicate configs."""
    configs = []
    for profile, alpha, split in (
        ("explore_alpha92p5_m005", 92.5, -5.0),
        ("explore_alpha92p5_m010", 92.5, -10.0),
        ("safe_alpha95_m000", 95.0, 0.0),
        ("aggressive_alpha90_m010", 90.0, -10.0),
    ):
        configs.append(
            _bundle_duplicate_config(
                family="bundle_split_duplicate",
                name=profile,
                primary_kind="bundle_split",
                alpha=alpha,
                high=0.92,
                mid=0.90,
                max_extra=30,
                max_couriers=3,
                min_roi=0.0,
                primary_params={"split_min_improvement": split},
                control_params={"split_min_improvement": split, "max_split_passes": 1},
            )
        )
    return configs


def build_bundle_merge_duplicate_configs() -> list[StrategyConfig]:
    """Build bundle-merge primary-search duplicate configs."""
    configs = []
    for profile, alpha, merge in (
        ("bundle_merge_focused", 90.0, 0.0),
        ("explore_alpha92p5_m010", 92.5, -10.0),
        ("explore_alpha92p5_m030", 92.5, -30.0),
        ("safe_alpha95_m000", 95.0, 0.0),
        ("scarce_bundle_bonus", 92.5, -10.0),
    ):
        configs.append(
            _bundle_duplicate_config(
                family="bundle_merge_duplicate",
                name=profile,
                primary_kind="bundle_merge",
                alpha=alpha,
                high=0.92,
                mid=0.90,
                max_extra=30,
                max_couriers=3,
                min_roi=0.0,
                primary_params={
                    "merge_min_improvement": merge,
                    "scarce_bundle_bonus": 60.0,
                },
                control_params={"merge_min_improvement": merge, "max_merge_passes": 1},
            )
        )
    return configs


def build_pressure_targeted_configs() -> list[StrategyConfig]:
    """Build structural pressure probes for scarce and low-willingness cases."""
    return [
        _base_config(
            name="bundle_first_b050",
            family="pressure_targeted",
            intent="scarce_courier_bundle_pressure",
            primary_kind="bundle_merge",
            primary_params={
                "alpha": 92.5,
                "scarce_alpha": 95.0,
                "scarce_bundle_bonus": 50.0,
                "scarce_ratio_threshold": 1.35,
                "scarce_use_merge": False,
                "merge_min_improvement": 0.0,
            },
            repairs=(("risk_tier_duplicate", _risk_params(0.95, 0.92, 30, 3, -15.0)),),
            control=_control(max_extra=30, max_couriers=3, min_roi=-15.0),
            tags=("scarce_courier_pressure", "bundle_underuse"),
        ),
        _base_config(
            name="scarce_b050_lowwill_c4",
            family="pressure_targeted",
            intent="combined_scarce_low_willingness_pressure",
            primary_kind="bundle_merge",
            primary_params={
                "alpha": 92.5,
                "low_willingness_alpha": 90.0,
                "scarce_alpha": 95.0,
                "scarce_bundle_bonus": 50.0,
                "scarce_ratio_threshold": 1.35,
            },
            repairs=(("risk_tier_duplicate", _risk_params(0.98, 0.95, 60, 4, -45.0)),),
            control=_control(max_extra=60, max_couriers=4, min_roi=-45.0),
            tags=("low_expected_success", "scarce_courier_pressure"),
        ),
        _base_config(
            name="pressure_balanced",
            family="pressure_targeted",
            intent="regret_rank_pressure_probe",
            primary_kind="regret_rank",
            primary_params={
                "alpha": 92.5,
                "regret_score_weight": 0.28,
                "regret_scarcity_weight": 90.0,
                "regret_willingness_weight": 35.0,
                "regret_bundle_bonus": 12.0,
            },
            repairs=(("risk_tier_duplicate", _risk_params(0.96, 0.93, 45, 3, -25.0)),),
            control=_control(max_extra=45, max_couriers=3, min_roi=-25.0),
            tags=("scarce_courier_pressure",),
        ),
        _base_config(
            name="drop_tail_05",
            family="pressure_targeted",
            intent="tail_drop_probe",
            primary_kind="regret_rank",
            primary_params={
                "alpha": 92.5,
                "regret_score_weight": 0.25,
                "regret_scarcity_weight": 80.0,
                "regret_willingness_weight": 35.0,
            },
            repairs=(("drop_tail_probe", {"drop_ratio": 0.05, "max_drop_tasks": 2, "min_task_count": 25}),),
            control=_control(max_extra=0, max_couriers=1, min_roi=0.0),
            tags=("pressure_probe",),
        ),
    ]


def build_portfolio_overlay_configs() -> list[StrategyConfig]:
    """Build portfolio and task-overlay probes for pressure-case robustness."""
    return [
        _portfolio_config("c4_task_overlay_balanced", 60, 4, -35.0, 45, 3, -20.0, 125.0),
        _portfolio_config("c4_task_overlay_deep", 70, 4, -45.0, 60, 4, -35.0, 150.0),
        _portfolio_config("c4_task_overlay_safe", 45, 3, -20.0, 30, 3, -5.0, 115.0),
        _portfolio_config("c3_repaired_overlay", 50, 3, -25.0, 35, 3, -5.0, 115.0),
    ]


def build_beam_staged_configs() -> list[StrategyConfig]:
    """Build beam-search and staged low-willingness pressure probes."""
    return [
        _beam_config("c4_beam_balanced", 10, 4, 420, "mixed", "scarce_and_extreme", 60, 4, -35.0, 45, 3, 0.95),
        _beam_config("c4_beam_deep", 16, 5, 560, "mixed", "scarce_and_extreme", 70, 4, -45.0, 60, 4, 0.97),
        _beam_config("c4_lowwill_stage_safe", 8, 3, 360, "mixed", "none", 50, 3, -25.0, 35, 3, 0.94),
        _beam_config("c4_scarce_bundle_beam", 14, 5, 520, "bundle_bias", "scarce_only", 45, 3, -25.0, 30, 3, 0.94),
    ]


DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT = "experiments/generated_variants"

STRATEGY_SPACE_REGISTRY: dict[str, StrategySpaceDefinition] = {
    "broad_strategy": StrategySpaceDefinition(
        name="broad_strategy",
        label="broad-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/broad_strategy",
        notes="Generated broad multi-family strategy configs.",
        build_configs=build_broad_strategy_configs,
        description="Bootstrap sweep across greedy objectives, score/probability weights, bundle bias, and basic local improvement.",
        targets=("bootstrap", "unknown"),
        risk_level="low",
        param_hints=("profile=broad", "optional alphas list for broad alpha probing"),
    ),
    "local_improve": StrategySpaceDefinition(
        name="local_improve",
        label="local-improve-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/local_improve",
        notes="Generated local-improvement neighborhood configs.",
        build_configs=build_local_improve_configs,
        description="Bounded replacement search around greedy assignments to reduce score cost while preserving validity.",
        targets=("high_score_cost", "invalid_or_timeout", "stagnation"),
        risk_level="low",
        param_hints=("alpha_min/alpha_max", "max_passes_min/max_passes_max"),
    ),
    "duplicate_augment": StrategySpaceDefinition(
        name="duplicate_augment",
        label="duplicate-augment-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/duplicate_augment",
        notes="Generated duplicate-dispatch augmentation configs.",
        build_configs=build_duplicate_augment_configs,
        description="Adds extra couriers to low-success assignments after a primary greedy/local assignment.",
        targets=("low_expected_success",),
        risk_level="medium",
        param_hints=("alpha_min/alpha_max", "max_extra_dispatches_min/max_extra_dispatches_max", "max_couriers_per_assignment_min/max"),
    ),
    "duplicate_augment_refine": StrategySpaceDefinition(
        name="duplicate_augment_refine",
        label="duplicate-augment-refine-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/duplicate_augment_refine",
        notes="Generated duplicate-dispatch refinement configs.",
        build_configs=build_duplicate_augment_refine_configs,
        description="Narrow duplicate-dispatch refinement around conservative extra-courier settings.",
        targets=("over_duplicate_cost", "low_expected_success"),
        risk_level="low",
        param_hints=("max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max"),
    ),
    "risk_tier_duplicate": StrategySpaceDefinition(
        name="risk_tier_duplicate",
        label="risk-tier-duplicate-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/risk_tier_duplicate",
        notes="Generated risk-tier duplicate-dispatch configs.",
        build_configs=build_risk_tier_duplicate_configs,
        description="Risk-tiered duplicate dispatch focused on tasks with low combined success probability.",
        targets=("low_expected_success",),
        risk_level="medium",
        param_hints=("high_risk_target_min/max", "max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max"),
    ),
    "low_willingness_deep_duplicate": StrategySpaceDefinition(
        name="low_willingness_deep_duplicate",
        label="low-willingness-deep-duplicate-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/low_willingness_deep_duplicate",
        notes="Generated low-willingness deep duplicate-dispatch probes.",
        build_configs=build_low_willingness_deep_duplicate_configs,
        description="Aggressive duplicate-dispatch sweep for persistent low-willingness and low-success cases.",
        targets=("low_expected_success",),
        risk_level="high",
        param_hints=("max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max", "min_roi_min/max"),
    ),
    "task_risk_duplicate": StrategySpaceDefinition(
        name="task_risk_duplicate",
        label="task-risk-duplicate-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/task_risk_duplicate",
        notes="Generated task-level duplicate-dispatch overlay configs.",
        build_configs=build_task_risk_duplicate_configs,
        description="Task-level risk overlay that can add duplicate couriers based on per-task success gaps.",
        targets=("low_expected_success", "bundle_underuse"),
        risk_level="medium",
        param_hints=("high_risk_target_min/max", "max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max"),
    ),
    "bundle_split_duplicate": StrategySpaceDefinition(
        name="bundle_split_duplicate",
        label="bundle-split-duplicate-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/bundle_split_duplicate",
        notes="Generated bundle-split primary-search duplicate configs.",
        build_configs=build_bundle_split_duplicate_configs,
        description="Tests splitting bundled assignments before duplicate repair when bundle choices look too costly.",
        targets=("high_score_cost", "over_duplicate_cost"),
        risk_level="medium",
        param_hints=("split_min_improvement_values", "max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max"),
    ),
    "bundle_merge_duplicate": StrategySpaceDefinition(
        name="bundle_merge_duplicate",
        label="bundle-merge-duplicate-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/bundle_merge_duplicate",
        notes="Generated bundle-merge primary-search duplicate configs.",
        build_configs=build_bundle_merge_duplicate_configs,
        description="Merges compatible single-task assignments to reduce courier pressure before duplicate repair.",
        targets=("scarce_courier_pressure", "bundle_underuse"),
        risk_level="medium",
        param_hints=("merge_min_improvement_values", "max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max"),
    ),
    "pressure_targeted": StrategySpaceDefinition(
        name="pressure_targeted",
        label="pressure-targeted-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/pressure_targeted",
        notes="Generated pressure-targeted probes for scarce couriers and low willingness.",
        build_configs=build_pressure_targeted_configs,
        description="Mixed structural probes for scarce couriers and low-willingness pressure cases.",
        targets=("scarce_courier_pressure", "low_expected_success", "bundle_underuse"),
        risk_level="high",
        param_hints=("max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max", "min_roi_min/max"),
    ),
    "portfolio_overlay": StrategySpaceDefinition(
        name="portfolio_overlay",
        label="portfolio-overlay-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/portfolio_overlay",
        notes="Generated portfolio and task-level overlay probes for scarce and low-willingness pressure cases.",
        build_configs=build_portfolio_overlay_configs,
        description="Portfolio-style overlays that combine scarce-case and task-level duplicate behaviors.",
        targets=("scarce_courier_pressure", "low_expected_success"),
        risk_level="high",
        param_hints=("max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max", "min_roi_min/max"),
    ),
    "beam_staged": StrategySpaceDefinition(
        name="beam_staged",
        label="beam-staged-strategy-sweep",
        output_dir=f"{DEFAULT_STRATEGY_SPACE_OUTPUT_ROOT}/beam_staged",
        notes="Generated beam-search primary assignment and staged low-willingness duplicate-dispatch probes.",
        build_configs=build_beam_staged_configs,
        description="Beam-search primary assignment followed by staged duplicate repair for structural pressure cases.",
        targets=("scarce_courier_pressure", "low_expected_success", "stagnation"),
        risk_level="high",
        param_hints=("max_extra_dispatches_min/max", "max_couriers_per_assignment_min/max"),
    ),
}


def list_strategy_space_names() -> list[str]:
    """Return registered local strategy spaces in deterministic order."""
    return list(STRATEGY_SPACE_REGISTRY)


def strategy_space_catalog() -> list[dict[str, Any]]:
    """Return a compact LLM-readable catalog of executable strategy spaces."""
    catalog = []
    for definition in STRATEGY_SPACE_REGISTRY.values():
        configs = definition.build_configs()
        catalog.append(
            {
                "name": definition.name,
                "description": definition.description or definition.notes,
                "targets": list(definition.targets),
                "risk_level": definition.risk_level,
                "param_hints": list(definition.param_hints),
                "seed_config_count": len(configs),
                "seed_configs": [
                    _strategy_config_catalog_entry(config) for config in configs
                ],
                "key_param_values": _catalog_key_param_values(configs),
            }
        )
    return catalog


def get_strategy_space_definition(name: str) -> StrategySpaceDefinition:
    """Return one registered strategy-space definition."""
    try:
        return STRATEGY_SPACE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_strategy_space_names())
        raise ValueError(
            f"unknown strategy space {name!r}; available: {available}"
        ) from exc


def build_strategy_configs(
    name: str,
    *,
    alphas: Sequence[float] | None = None,
    selected_config_ids: Sequence[str] | None = None,
    inline_configs: Sequence[dict[str, Any] | StrategyConfig] | None = None,
) -> list[StrategyConfig]:
    """Build configs for one registered local strategy space."""
    if name == "broad_strategy" and alphas is not None:
        configs = build_broad_strategy_configs(alphas=alphas)
    else:
        configs = get_strategy_space_definition(name).build_configs()
    configs = [*configs, *[_coerce_strategy_config(config) for config in inline_configs or []]]
    selected = {str(config_id) for config_id in selected_config_ids or [] if config_id}
    if selected:
        return [config for config in configs if config.config_id in selected]
    return configs


def strategy_config_ids_for_region(
    name: str,
    param_region: dict[str, Any] | None,
    *,
    fallback_to_all: bool = True,
) -> list[str]:
    """Return config IDs matching an evidence-selected region."""
    configs = get_strategy_space_definition(name).build_configs()
    if not param_region:
        return [config.config_id for config in configs]
    matched = [
        config.config_id
        for config in configs
        if _config_matches_region(config, param_region)
    ]
    if matched or not fallback_to_all:
        return matched
    return [config.config_id for config in configs]


def strategy_sweep_label(name: str) -> str:
    """Return the append-only experiment label for one strategy space."""
    return get_strategy_space_definition(name).label


def default_strategy_output_dir(name: str) -> str:
    """Return the default materialization directory for one strategy space."""
    return get_strategy_space_definition(name).output_dir


def strategy_sweep_notes(name: str, *, hypothesis: str = "") -> str:
    """Return append-only experiment notes for one strategy sweep."""
    base = get_strategy_space_definition(name).notes
    if hypothesis:
        return f"{base} Hypothesis: {hypothesis}"
    return base


def render_solver(config: StrategyConfig) -> str:
    """Render a Python 3.6-compatible standalone solver for one config."""
    return render_solver_source(strategy_config_to_dict(config))


def materialize_strategy_configs(
    configs: Sequence[StrategyConfig],
    output_dir: str | Path,
) -> list[Path]:
    """Write generated strategy configs and return their variant paths."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for config in configs:
        path = root / strategy_variant_filename(config)
        path.write_text(render_solver(config), encoding="utf-8")
        paths.append(path)
    return paths


def strategy_variant_filename(config: StrategyConfig) -> str:
    """Return a stable filename for one generated strategy config."""
    family = _safe_token(config.family)
    name = _safe_token(config.name)
    return f"{family}_{name}_{config.config_id[:12]}.py"


def read_strategy_config_header(path: str | Path) -> StrategyConfig:
    """Read a StrategyConfig header from a generated solver file."""
    with Path(path).open(encoding="utf-8") as file:
        first_line = file.readline().strip()
    if not first_line.startswith(CONFIG_HEADER_PREFIX):
        raise ValueError(f"missing StrategyConfig header in {path}")
    payload = json.loads(first_line[len(CONFIG_HEADER_PREFIX) :])
    if not isinstance(payload, dict):
        raise ValueError("StrategyConfig header must contain an object")
    return strategy_config_from_dict(payload)


def parse_strategy_metadata_from_variant_path(path: str | Path) -> dict[str, Any]:
    """Return strategy metadata from a generated solver StrategyConfig header."""
    try:
        config = read_strategy_config_header(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "family": "unknown",
            "intent": "unknown",
            "pipeline": "unknown",
            "params": {},
            "config_id": "",
            "config": {},
        }
    return {
        "family": config.family,
        "intent": config.intent,
        "pipeline": config.pipeline,
        "params": strategy_config_key_params(config),
        "config_id": config.config_id,
        "config": strategy_config_to_dict(config),
    }


def strategy_search_leaderboard_rows(
    batch: VariantSuiteBatchResult,
) -> list[dict[str, Any]]:
    """Return compact strategy leaderboard rows."""
    return [_strategy_result_row(result) for result in batch.results]


def select_strategy_solver_candidates(
    batch: VariantSuiteBatchResult,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Select diverse solver candidates from the strategy ranking."""
    rows = [row for row in strategy_search_leaderboard_rows(batch) if row["is_valid"]]
    selections: list[dict[str, Any]] = []
    _append_strategy_selection(selections, rows[0] if rows else None, "best_proxy")
    for reason, predicate in (
        ("low_risk", lambda row: row.get("risk_level") == "low"),
        ("bundle_probe", lambda row: "bundle" in str(row.get("pipeline", ""))),
        ("duplicate_probe", lambda row: "duplicate" in str(row.get("pipeline", ""))),
        ("task_overlay_probe", lambda row: "task_overlay" in str(row.get("pipeline", ""))),
        ("beam_probe", lambda row: "beam" in str(row.get("pipeline", ""))),
    ):
        _append_strategy_selection(
            selections,
            next((row for row in rows if predicate(row)), None),
            reason,
        )
        if len(selections) >= limit:
            return selections[:limit]
    for row in rows:
        _append_strategy_selection(selections, row, "next_best_proxy")
        if len(selections) >= limit:
            break
    return selections[:limit]


def validate_strategy_config(config: StrategyConfig) -> None:
    """Validate semantic renderability for one strategy config."""
    seen_candidate_phase = False
    for repair in config.repairs:
        if repair.kind == "local_improve":
            if seen_candidate_phase:
                raise ValueError("local_improve must run before assignment repairs")
            continue
        seen_candidate_phase = True
    render_solver(config)


def strategy_config_key_params(config: StrategyConfig) -> dict[str, float | int | str | bool]:
    """Return compact decision-relevant params for one config."""
    params: dict[str, float | int | str | bool] = {}
    params.update(config.primary.params)
    for repair in config.repairs:
        params.update(repair.params)
    control = strategy_config_to_dict(config)["control"]
    for key in (
        "max_extra_dispatches",
        "max_couriers_per_assignment",
        "min_roi",
        "high_risk_target",
        "mid_risk_target",
        "time_budget_seconds",
    ):
        value = control.get(key)
        if value is not None:
            params[key] = value
    params.update(config.control.params)
    params["name"] = config.name
    params["family"] = config.family
    return {
        key: value
        for key, value in sorted(params.items())
        if key in CATALOG_PARAM_KEYS or key in {"name", "family"}
    }


def _base_config(
    *,
    name: str,
    family: str,
    intent: str,
    primary_kind: str,
    primary_params: dict[str, float | int | str | bool] | None = None,
    repairs: Sequence[tuple[str, dict[str, float | int | str | bool]]] = (),
    control: ControlConfig | None = None,
    tags: Sequence[str] = (),
) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        family=family,
        intent=intent,
        source="seed_config",
        primary=PrimaryConfig(primary_kind, primary_params or {}),
        repairs=tuple(RepairStep(kind, params) for kind, params in repairs),
        control=control or ControlConfig(time_budget_seconds=8.5),
        tags=tuple(tags),
    )


def _local_improve_config(alpha: float, max_passes: int, time_budget: float) -> StrategyConfig:
    return _base_config(
        name=(
            "local_improve_"
            f"{_format_param_token('a', alpha)}_"
            f"{_format_param_token('p', float(max_passes))}_"
            f"{_format_param_token('t', time_budget)}"
        ),
        family="local_improve",
        intent="score_cost_local_replacement",
        primary_kind="willingness_adjusted",
        primary_params={"alpha": float(alpha)},
        repairs=(("local_improve", {"max_passes": int(max_passes), "time_budget_seconds": float(time_budget)}),),
        control=ControlConfig(time_budget_seconds=float(time_budget)),
        tags=("high_score_cost",),
    )


def _duplicate_config(
    *,
    family: str,
    name_prefix: str,
    alpha: float,
    min_success: float,
    min_roi: float,
    max_extra: int,
    max_couriers: int,
) -> StrategyConfig:
    return _base_config(
        name=(
            f"{name_prefix}_"
            f"{_format_param_token('a', alpha)}_"
            f"{_format_param_token('s', min_success)}_"
            f"{_format_signed_param_token('rp', min_roi)}_"
            f"{_format_param_token('m', float(max_extra))}_"
            f"{_format_param_token('c', float(max_couriers))}"
        ),
        family=family,
        intent="duplicate_dispatch_success_repair",
        primary_kind="willingness_adjusted",
        primary_params={"alpha": float(alpha)},
        repairs=(
            ("local_improve", {"max_passes": 1}),
            (
                "duplicate_dispatch",
                {
                    "min_success_probability": float(min_success),
                    "min_roi": float(min_roi),
                    "max_extra_dispatches": int(max_extra),
                    "max_couriers_per_assignment": int(max_couriers),
                    "expected_failed_weight": SOLVER_EXPECTED_FAILED_WEIGHT,
                    "expected_success_credit": SOLVER_EXPECTED_SUCCESS_CREDIT,
                    "score_weight": SOLVER_SCORE_WEIGHT,
                },
            ),
        ),
        control=_control(max_extra=max_extra, max_couriers=max_couriers, min_roi=min_roi),
        tags=("low_expected_success",),
    )


def _risk_params(
    high: float,
    mid: float,
    max_extra: int,
    max_couriers: int,
    min_roi: float,
    *,
    high_success_min_roi: float = 30.0,
) -> dict[str, float | int | str | bool]:
    return {
        "high_risk_target": float(high),
        "mid_risk_target": float(mid),
        "max_extra_dispatches": int(max_extra),
        "max_couriers_per_assignment": int(max_couriers),
        "min_roi": float(min_roi),
        "high_success_min_roi": float(high_success_min_roi),
        "score_weight": SOLVER_SCORE_WEIGHT,
        "expected_failed_weight": SOLVER_EXPECTED_FAILED_WEIGHT,
        "expected_success_credit": SOLVER_EXPECTED_SUCCESS_CREDIT,
    }


def _risk_config(
    family: str,
    profile: str,
    alpha: float,
    high: float,
    mid: float,
    max_extra: int,
    max_couriers: int,
    *,
    min_roi: float = 0.0,
    high_success_min_roi: float = 30.0,
    low_willingness_alpha: float | None = None,
) -> StrategyConfig:
    primary_params: dict[str, float | int | str | bool] = {"alpha": float(alpha)}
    if low_willingness_alpha is not None:
        primary_params["low_willingness_alpha"] = float(low_willingness_alpha)
    return _base_config(
        name=(
            f"{profile}_"
            f"{_format_param_token('a', alpha)}_"
            f"{_format_param_token('h', high)}_"
            f"{_format_param_token('d', mid)}_"
            f"{_format_param_token('x', float(max_extra))}_"
            f"{_format_param_token('c', float(max_couriers))}"
        ),
        family=family,
        intent="risk_tier_duplicate_success_repair",
        primary_kind="willingness_adjusted",
        primary_params=primary_params,
        repairs=(
            ("local_improve", {"max_passes": 1}),
            ("risk_tier_duplicate", _risk_params(high, mid, max_extra, max_couriers, min_roi, high_success_min_roi=high_success_min_roi)),
        ),
        control=_control(max_extra=max_extra, max_couriers=max_couriers, min_roi=min_roi, high=high, mid=mid),
        tags=("low_expected_success", profile),
    )


def _task_overlay_config(
    profile: str,
    alpha: float,
    high: float,
    mid: float,
    max_extra: int,
    max_couriers: int,
    min_roi: float,
    *,
    bundle_bias: float = 0.0,
) -> StrategyConfig:
    return _base_config(
        name=(
            f"{profile}_"
            f"{_format_param_token('a', alpha)}_"
            f"{_format_param_token('h', high)}_"
            f"{_format_param_token('d', mid)}_"
            f"{_format_param_token('x', float(max_extra))}_"
            f"{_format_param_token('c', float(max_couriers))}"
        ),
        family="task_risk_duplicate",
        intent="task_overlay_success_repair",
        primary_kind="willingness_adjusted",
        primary_params={"alpha": float(alpha), "bundle_bias": float(bundle_bias)},
        repairs=(
            ("local_improve", {"max_passes": 1}),
            ("task_overlay", _risk_params(high, mid, max_extra, max_couriers, min_roi)),
        ),
        control=_control(max_extra=max_extra, max_couriers=max_couriers, min_roi=min_roi, high=high, mid=mid),
        tags=("low_expected_success", "task_overlay", profile),
    )


def _bundle_duplicate_config(
    *,
    family: str,
    name: str,
    primary_kind: str,
    alpha: float,
    high: float,
    mid: float,
    max_extra: int,
    max_couriers: int,
    min_roi: float,
    primary_params: dict[str, float | int | str | bool],
    control_params: dict[str, float | int | str | bool],
) -> StrategyConfig:
    params = {"alpha": float(alpha), **primary_params}
    return _base_config(
        name=(
            f"{name}_"
            f"{_format_param_token('a', alpha)}_"
            f"{_format_param_token('h', high)}_"
            f"{_format_param_token('d', mid)}_"
            f"{_format_param_token('x', float(max_extra))}_"
            f"{_format_param_token('c', float(max_couriers))}"
        ),
        family=family,
        intent=f"{primary_kind}_with_duplicate_repair",
        primary_kind=primary_kind,
        primary_params=params,
        repairs=(
            ("local_improve", {"max_passes": 1}),
            ("risk_tier_duplicate", _risk_params(high, mid, max_extra, max_couriers, min_roi)),
        ),
        control=_control(max_extra=max_extra, max_couriers=max_couriers, min_roi=min_roi, high=high, mid=mid, params=control_params),
        tags=("bundle_underuse", "scarce_courier_pressure", name),
    )


def _portfolio_config(
    profile: str,
    max_extra: int,
    max_couriers: int,
    min_roi: float,
    overlay_extra: int,
    overlay_couriers: int,
    overlay_min_roi: float,
    success_credit: float,
) -> StrategyConfig:
    return _base_config(
        name=profile,
        family="portfolio_overlay",
        intent="portfolio_task_overlay_pressure_repair",
        primary_kind="bundle_merge",
        primary_params={
            "alpha": 92.5,
            "low_willingness_alpha": 90.0,
            "scarce_alpha": 95.0,
            "scarce_bundle_bonus": 50.0,
            "scarce_ratio_threshold": 1.35,
        },
        repairs=(
            ("risk_tier_duplicate", _risk_params(0.98, 0.95, max_extra, max_couriers, min_roi)),
            (
                "task_overlay",
                {
                    "high_risk_target": 0.98,
                    "mid_risk_target": 0.95,
                    "max_extra_dispatches": int(overlay_extra),
                    "max_couriers_per_assignment": int(overlay_couriers),
                    "min_roi": float(overlay_min_roi),
                    "portfolio_success_credit": float(success_credit),
                    "expected_success_credit": float(success_credit),
                },
            ),
        ),
        control=_control(max_extra=max_extra, max_couriers=max_couriers, min_roi=min_roi, high=0.98, mid=0.95),
        tags=("scarce_courier_pressure", "low_expected_success", "task_overlay"),
    )


def _beam_config(
    profile: str,
    width: int,
    top_per_task: int,
    global_limit: int,
    mode: str,
    scope: str,
    max_extra: int,
    max_couriers: int,
    min_roi: float,
    staged_extra: int,
    staged_couriers: int,
    staged_target: float,
) -> StrategyConfig:
    return _base_config(
        name=profile,
        family="beam_staged",
        intent="beam_primary_staged_duplicate_repair",
        primary_kind="beam",
        primary_params={
            "alpha": 92.5,
            "low_willingness_alpha": 90.0,
            "beam_width": int(width),
            "beam_top_per_task": int(top_per_task),
            "beam_global_limit": int(global_limit),
            "beam_success_credit": 112.0,
            "beam_pair_penalty": 1.5,
            "beam_bundle_penalty": 0.0,
            "beam_mode": mode,
            "beam_scope": scope,
        },
        repairs=(
            ("risk_tier_duplicate", _risk_params(0.98, 0.95, max_extra, max_couriers, min_roi)),
            (
                "staged_duplicate",
                {
                    "staged_max_extra_dispatches": int(staged_extra),
                    "staged_max_couriers": int(staged_couriers),
                    "staged_tail_fraction": 0.45,
                    "staged_tail_success_threshold": 0.90,
                    "staged_target": float(staged_target),
                    "staged_min_roi": float(min_roi),
                    "max_extra_dispatches": int(staged_extra),
                    "max_couriers_per_assignment": int(staged_couriers),
                    "min_roi": float(min_roi),
                    "high_risk_target": float(staged_target),
                    "mid_risk_target": min(float(staged_target), 0.95),
                },
            ),
        ),
        control=_control(max_extra=max_extra, max_couriers=max_couriers, min_roi=min_roi, high=0.98, mid=0.95),
        tags=("beam", "staged_duplicate", "low_expected_success"),
    )


def _control(
    *,
    max_extra: int,
    max_couriers: int,
    min_roi: float,
    high: float | None = None,
    mid: float | None = None,
    params: dict[str, float | int | str | bool] | None = None,
) -> ControlConfig:
    return ControlConfig(
        max_extra_dispatches=int(max_extra),
        max_couriers_per_assignment=int(max_couriers),
        min_roi=float(min_roi),
        high_risk_target=high,
        mid_risk_target=mid,
        time_budget_seconds=8.5,
        params=params or {},
    )


def _strategy_config_catalog_entry(config: StrategyConfig) -> dict[str, Any]:
    payload = strategy_config_to_dict(config)
    return {
        "config_id": config.config_id,
        "name": config.name,
        "family": config.family,
        "intent": config.intent,
        "pipeline": config.pipeline,
        "source": config.source,
        "primary": payload["primary"],
        "repairs": payload["repairs"],
        "control": payload["control"],
        "tags": list(config.tags),
        "key_params": strategy_config_key_params(config),
    }


def _catalog_key_param_values(
    configs: Sequence[StrategyConfig],
) -> dict[str, list[float | int | str | bool]]:
    values_by_key: dict[str, set[float | int | str | bool]] = {}
    for config in configs:
        for key, value in strategy_config_key_params(config).items():
            if key in {"name", "family"}:
                continue
            values_by_key.setdefault(key, set()).add(value)
    return {
        key: sorted(values, key=_catalog_sort_key)
        for key, values in sorted(values_by_key.items())
    }


def _strategy_result_row(result: Any) -> dict[str, Any]:
    aggregate = result.suite_result.aggregate_metrics
    metadata = parse_strategy_metadata_from_variant_path(result.variant_path)
    return {
        "rank": result.rank,
        "variant_path": result.variant_path,
        "config_id": metadata["config_id"],
        "family": metadata["family"],
        "intent": metadata["intent"],
        "pipeline": metadata["pipeline"],
        "params": metadata["params"],
        "is_valid": aggregate.is_valid,
        "timeout_count": aggregate.timeout_count,
        "invalid_case_count": aggregate.invalid_case_count,
        "mean_proxy_score": aggregate.mean_proxy_score,
        "mean_expected_success_ratio": aggregate.mean_expected_success_ratio,
        "mean_task_coverage_ratio": aggregate.mean_task_coverage_ratio,
        "mean_total_score": _mean(
            [
                case_result.metrics.total_score
                for case_result in result.suite_result.results
            ]
        ),
        "mean_duplicate_dispatch_assignment_count": _mean(
            [
                float(case_result.metrics.duplicate_dispatch_assignment_count)
                for case_result in result.suite_result.results
            ]
        ),
        "output_signature": _suite_output_signature(result.suite_result.results),
        "worst_case_id": aggregate.worst_case_id,
        "risk_level": _risk_level_for_family(metadata["family"]),
    }


def _append_strategy_selection(
    selections: list[dict[str, Any]],
    row: dict[str, Any] | None,
    reason: str,
) -> None:
    if row is None:
        return
    if any(selection["variant_path"] == row["variant_path"] for selection in selections):
        return
    if any(
        selection.get("output_signature") == row.get("output_signature")
        for selection in selections
    ):
        return
    selected = dict(row)
    selected["reason"] = reason
    selections.append(selected)


def _config_matches_region(config: StrategyConfig, region: dict[str, Any]) -> bool:
    params = strategy_config_key_params(config)
    profile = str(region.get("profile", ""))
    if profile and profile not in {"broad", "continue", "v3_pressure_sequence"}:
        supported = {config.name, config.family, *config.tags, config.intent}
        if profile not in supported:
            profile_like = {
                "risk_focused": {"risk_tier_duplicate", "low_expected_success"},
                "low_willingness_deep": {"low_willingness_deep_duplicate"},
                "bundle_merge_focused": {"bundle_merge_duplicate", "bundle_underuse"},
                "conservative_duplicate": {"conservative", "duplicate_augment_refine"},
                "score_cost_local_improve": {"local_improve"},
                "low_complexity": {"local_improve", "duplicate_augment"},
            }
            if not (profile_like.get(profile, set()) & supported):
                return False
    strategy_family = str(region.get("strategy_family", ""))
    if strategy_family and strategy_family not in {config.family, config.intent, *config.tags}:
        return False
    for key in (
        "alpha",
        "high_risk_target",
        "max_extra_dispatches",
        "max_couriers_per_assignment",
        "min_roi",
        "max_passes",
        "time_budget_seconds",
    ):
        if not _param_in_range(params, key, region):
            return False
    for key in ("merge_min_improvement", "split_min_improvement"):
        values_key = f"{key}_values"
        if values_key in region and key in params:
            allowed = {float(value) for value in region[values_key]}
            if float(params[key]) not in allowed:
                return False
    return True


def _param_in_range(
    params: dict[str, float | int | str | bool],
    name: str,
    region: dict[str, Any],
) -> bool:
    value = params.get(name)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return True
    lower_key = f"{name}_min"
    upper_key = f"{name}_max"
    if lower_key in region and float(value) < float(region[lower_key]):
        return False
    if upper_key in region and float(value) > float(region[upper_key]):
        return False
    return True


def _coerce_strategy_config(config: dict[str, Any] | StrategyConfig) -> StrategyConfig:
    if isinstance(config, StrategyConfig):
        return config
    return strategy_config_from_dict(config)


def _validate_param_keys(
    kind: str,
    params: dict[str, Any],
    allowed_by_kind: dict[str, set[str]],
) -> None:
    unknown = set(params) - allowed_by_kind[kind]
    if unknown:
        raise ValueError(
            f"unknown {kind} param(s): {', '.join(sorted(str(key) for key in unknown))}"
        )


def _validate_probability(field_name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected integer-compatible value")
    if int(value) != float(value):
        raise ValueError("expected integer value")
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric value")
    return float(value)


def _parse_scalar_dict(value: Any) -> dict[str, float | int | str | bool]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected object of scalar values")
    result: dict[str, float | int | str | bool] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            result[str(key)] = item
        elif isinstance(item, (float, int, str)):
            result[str(key)] = item
        else:
            raise ValueError(f"unsupported non-scalar config value for {key}")
    return dict(sorted(result.items()))


def _canonical_scalar_dict(
    value: dict[str, float | int | str | bool],
) -> dict[str, float | int | str | bool]:
    return dict(sorted(_parse_scalar_dict(value).items()))


def _canonical_jsonish_dict(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        if isinstance(item, dict):
            result[str(key)] = _canonical_jsonish_dict(item)
        elif isinstance(item, list):
            result[str(key)] = [
                _canonical_jsonish_dict(entry) if isinstance(entry, dict) else entry
                for entry in item
            ]
        elif isinstance(item, tuple):
            result[str(key)] = [
                _canonical_jsonish_dict(entry) if isinstance(entry, dict) else entry
                for entry in item
            ]
        else:
            result[str(key)] = item
    return result


def _format_param_token(prefix: str, value: float) -> str:
    scaled = int(round(float(value) * 10.0))
    whole = scaled // 10
    decimal = abs(scaled % 10)
    if decimal == 0:
        return f"{prefix}{whole:03d}"
    return f"{prefix}{whole:03d}p{decimal}"


def _format_signed_param_token(prefix: str, value: float) -> str:
    sign = "m" if value < 0 else "p"
    return f"{prefix}{sign}{_format_param_token('', abs(value))}"


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "strategy"


def _catalog_sort_key(value: float | int | str | bool) -> tuple[str, str]:
    if isinstance(value, bool):
        return ("bool", str(value))
    if isinstance(value, (float, int)):
        return ("number", f"{float(value):.12g}")
    return ("string", str(value))


def _risk_level_for_family(family: str) -> str:
    try:
        return get_strategy_space_definition(family).risk_level
    except ValueError:
        return "medium"


def _suite_output_signature(results: list[Any]) -> str:
    parts = []
    for result in results:
        output = getattr(result, "output", None)
        if output is None:
            output = getattr(result, "suite_result", None)
        parts.append(json.dumps(output, sort_keys=True, default=str))
    encoded = "\n".join(parts).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
