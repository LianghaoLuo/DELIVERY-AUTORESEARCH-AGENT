"""Generate and evaluate local strategy-space solver variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoresearch_agent.research.experiment_store import ExperimentStore
from autoresearch_agent.research.strategy_space import (
    build_strategy_configs,
    default_strategy_output_dir,
    list_strategy_space_names,
    materialize_strategy_configs,
    parse_alpha_list,
    strategy_search_leaderboard_rows,
    strategy_sweep_label,
    strategy_sweep_notes,
)
from autoresearch_agent.solver_dev.case_suite import build_case_suite
from autoresearch_agent.solver_dev.runner import run_solver_suite
from autoresearch_agent.solver_dev.variants import run_solver_variant_suite_batch

DEFAULT_ALPHAS = "70,75,80,85,90,92.5,95,100,110,125"
DEFAULT_OUTPUT_DIR = ""


def main() -> None:
    """Run a local strategy-space sweep."""
    args = _parse_args()
    alphas = (
        parse_alpha_list(args.alphas) if args.search_space == "broad_strategy" else None
    )
    configs = build_strategy_configs(args.search_space, alphas=alphas)
    output_dir = args.output_dir or default_strategy_output_dir(args.search_space)
    notes = (
        _build_notes(alphas=alphas)
        if args.search_space == "broad_strategy" and alphas is not None
        else strategy_sweep_notes(args.search_space)
    )
    variant_paths = materialize_strategy_configs(configs, output_dir)

    input_text = Path(args.data_file).read_text()
    cases = build_case_suite(
        input_text,
        source_name=args.data_file,
        suite_name=args.case_suite,
    )
    baseline = run_solver_suite(
        args.baseline_solver,
        cases,
        timeout_seconds=args.timeout_seconds,
    )
    batch = run_solver_variant_suite_batch(
        variant_paths,
        cases,
        timeout_seconds=args.timeout_seconds,
    )
    rows = strategy_search_leaderboard_rows(batch)
    rows = _rank_rows(rows, metric=args.ranking_metric)
    record = ExperimentStore(
        root_dir=Path(args.experiments_dir)
    ).append_variant_suite_batch(
        batch,
        label=strategy_sweep_label(args.search_space),
        baseline_result=baseline,
        notes=notes,
    )

    print("run_id:", record["run_id"])
    print("generated_configs:", len(variant_paths))
    print()
    print(
        _render_strategy_leaderboard(
            rows,
            limit=args.leaderboard_limit,
            ranking_metric=args.ranking_metric,
        )
    )
    print()
    print(
        _render_solver_candidates(
            _select_solver_candidates_from_rows(
                rows,
                limit=args.candidate_limit,
            )
        )
    )


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a local strategy-space sweep.")
    parser.add_argument(
        "--search-space",
        choices=list_strategy_space_names(),
        default="local_improve",
    )
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--data-file", default="data/large_seed301.txt")
    parser.add_argument("--baseline-solver", default="solvers/solver.py")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--case-suite",
        choices=["robust"],
        default="robust",
    )
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--leaderboard-limit", type=int, default=12)
    parser.add_argument(
        "--ranking-metric",
        choices=["mean_proxy_score"],
        default="mean_proxy_score",
    )
    return parser.parse_args()


def _render_strategy_leaderboard(
    rows: list[dict],
    *,
    limit: int,
    ranking_metric: str,
) -> str:
    """Render a compact strategy leaderboard."""
    lines = [
        f"Strategy leaderboard ranked by `{ranking_metric}`:",
        "| Rank | Family | Strategy | Params | Valid | Mean Proxy | Expected Success | Total Score | Dup Assign | Variant |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows[:limit]:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"`{row['family']}` | "
            f"`{row['pipeline']}` | "
            f"`{row['params']}` | "
            f"{row['is_valid']} | "
            f"{row['mean_proxy_score']:.6f} | "
            f"{row['mean_expected_success_ratio']:.6f} | "
            f"{row['mean_total_score']:.6f} | "
            f"{row['mean_duplicate_dispatch_assignment_count']:.3f} | "
            f"`{row['variant_path']}` |"
        )
    return "\n".join(lines)


def _rank_rows(rows: list[dict], *, metric: str) -> list[dict]:
    """Return valid-first rows ranked by the requested local metric."""
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["is_valid"]),
            float(row[metric]),
            str(row["variant_path"]),
        ),
    )
    result = []
    for rank, row in enumerate(ranked, start=1):
        copied = dict(row)
        copied["rank"] = rank
        result.append(copied)
    return result


def _select_solver_candidates_from_rows(
    rows: list[dict],
    *,
    limit: int,
) -> list[dict]:
    """Select diverse candidates from already-ranked strategy rows."""
    selections: list[dict] = []
    for reason, row in (
        ("best_ranked_proxy", rows[0] if rows else None),
        ("profile_conservative", _first_matching_row(rows, name_contains="conservative")),
        (
            "profile_low_willingness_aggressive",
            _first_matching_row(rows, name_contains="low_willingness_aggressive"),
        ),
        ("profile_scarce_safe", _first_matching_row(rows, name_contains="scarce_safe")),
        ("task_risk_balanced", _first_matching_row(rows, name_contains="balanced")),
        (
            "bundle_split_explore",
            _first_matching_row(rows, name_contains="explore_alpha92p5_m005"),
        ),
        (
            "bundle_merge_explore",
            _first_matching_row(rows, name_contains="explore_alpha92p5_m010"),
        ),
        (
            "task_risk_aggressive",
            _first_matching_row(rows, name_contains="low_willingness_aggressive"),
        ),
        ("conservative_roi", _first_matching_row(rows, min_roi=25.0)),
        ("aggressive_extra", _first_matching_row(rows, max_extra_dispatches=30)),
        ("three_courier_probe", _first_matching_row(rows, max_couriers=3)),
    ):
        _append_selection(selections, row, reason)
        if len(selections) >= limit:
            return selections
    for row in rows:
        _append_selection(selections, row, "next_best_ranked")
        if len(selections) >= limit:
            break
    return selections


def _first_matching_row(
    rows: list[dict],
    *,
    min_roi: float | None = None,
    max_extra_dispatches: int | None = None,
    max_couriers: int | None = None,
    name_contains: str | None = None,
) -> dict | None:
    """Return the first row matching requested duplicate-augment params."""
    for row in rows:
        params = row.get("params", {})
        if not isinstance(params, dict):
            continue
        if min_roi is not None and float(params.get("min_roi", 0.0)) != min_roi:
            continue
        if name_contains is not None and name_contains not in str(params.get("name", "")):
            continue
        if (
            max_extra_dispatches is not None
            and int(params.get("max_extra_dispatches", 0)) != max_extra_dispatches
        ):
            continue
        if (
            max_couriers is not None
            and int(params.get("max_couriers_per_assignment", 0)) != max_couriers
        ):
            continue
        return row
    return None


def _append_selection(
    selections: list[dict],
    row: dict | None,
    reason: str,
) -> None:
    """Append a row if it is valid and not already selected."""
    if row is None or not bool(row.get("is_valid", False)):
        return
    if any(
        selection["variant_path"] == row["variant_path"] for selection in selections
    ):
        return
    if any(
        selection.get("output_signature") == row.get("output_signature")
        for selection in selections
    ):
        return
    selected = dict(row)
    selected["reason"] = reason
    selections.append(selected)


def _render_solver_candidates(candidates: list[dict]) -> str:
    """Render recommended solver candidates from proxy ranking."""
    lines = [
        "Recommended solver candidates:",
        "| Reason | Rank | Family | Strategy | Mean Proxy | Output Sig | Variant |",
        "| --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    for candidate in candidates:
        lines.append(
            "| "
            f"`{candidate['reason']}` | "
            f"{candidate['rank']} | "
            f"`{candidate['family']}` | "
            f"`{candidate['pipeline']}` | "
            f"{candidate['mean_proxy_score']:.6f} | "
            f"`{candidate['output_signature']}` | "
            f"`{candidate['variant_path']}` |"
        )
    return "\n".join(lines)


def _build_notes(
    *,
    alphas: list[float],
) -> str:
    """Build append-only experiment notes for this sweep."""
    return (
        "Generated broad multi-family strategy configs with alphas "
        f"{','.join(_format_alpha(alpha) for alpha in alphas)}. "
        "Ranked by the default metrics.py proxy."
    )


def _format_alpha(alpha: float) -> str:
    """Return a compact alpha string."""
    return str(int(alpha)) if alpha.is_integer() else str(alpha)


if __name__ == "__main__":
    main()
