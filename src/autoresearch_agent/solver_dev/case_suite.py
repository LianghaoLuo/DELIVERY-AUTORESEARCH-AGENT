"""Deterministic local case-suite generation for solver evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from autoresearch_agent.solver_dev.parser import Candidate, parse_candidate_table

CaseSuiteName = Literal["robust"]


@dataclass(frozen=True)
class SolverCase:
    """One in-memory solver evaluation case."""

    case_id: str
    input_text: str
    source_name: str = ""
    weight: float = 1.0


def build_case_suite(
    input_text: str,
    source_name: str = "",
    *,
    suite_name: CaseSuiteName = "robust",
) -> list[SolverCase]:
    """Build one named deterministic local case suite."""
    if suite_name == "robust":
        return build_robust_case_suite(input_text, source_name=source_name)
    raise ValueError(f"unknown case suite: {suite_name}")


def build_robust_case_suite(
    input_text: str, source_name: str = ""
) -> list[SolverCase]:
    """Build topology-varied local cases from one candidate table.

    The robust suite keeps a full anchor case, then creates deterministic
    resamples that vary task/courier ratios, candidate density, and willingness
    tails without reading external score records.
    """
    table = parse_candidate_table(input_text)
    task_ids = sorted(table.task_ids)
    low_tasks = _select_low_willingness_tasks(table.candidates, limit=20)
    tail_tasks = _select_low_willingness_tasks(
        table.candidates,
        limit=max(1, len(task_ids) // 3),
    )

    medium_tasks = _select_hash_ranked_items(
        task_ids, limit=min(25, len(task_ids)), salt="robust_medium"
    )
    medium_sparse_tasks = _select_hash_ranked_items(
        task_ids, limit=min(25, len(task_ids)), salt="robust_medium_sparse"
    )
    small_tasks = _select_hash_ranked_items(
        task_ids, limit=min(10, len(task_ids)), salt="robust_small"
    )

    medium_candidates = _build_task_courier_subset(
        table.candidates,
        selected_task_ids=medium_tasks,
        courier_limit=50,
        salt="robust_medium_hash",
        per_task_candidate_limit=160,
    )
    medium_sparse_candidates = _build_task_courier_subset(
        table.candidates,
        selected_task_ids=medium_sparse_tasks,
        courier_limit=38,
        salt="robust_medium_sparse",
        keep_ratio=0.50,
        min_candidates_per_task=22,
        per_task_candidate_limit=80,
    )
    lowwill_candidates = _build_task_courier_subset(
        table.candidates,
        selected_task_ids=low_tasks,
        courier_limit=34,
        salt="robust_lowwill_sparse",
        keep_ratio=0.5,
        min_candidates_per_task=20,
        per_task_candidate_limit=72,
        score_jitter=0.04,
        willingness_jitter=0.02,
    )
    lowwill_candidates = _transform_low_willingness_tail(
        lowwill_candidates,
        low_task_ids=low_tasks,
        low_factor=0.58,
        other_factor=0.75,
        low_score_factor=1.04,
    )
    scarce_soft_candidates = _build_task_courier_subset(
        table.candidates,
        selected_task_ids=task_ids,
        courier_limit=30,
        salt="robust_scarce_soft",
        keep_ratio=0.66,
        min_candidates_per_task=34,
        per_task_candidate_limit=115,
    )
    scarce_neutral_candidates = _build_task_courier_subset(
        table.candidates,
        selected_task_ids=task_ids,
        courier_limit=26,
        salt="robust_scarce_neutral",
        keep_ratio=0.62,
        min_candidates_per_task=30,
        per_task_candidate_limit=96,
    )
    small_candidates = _build_task_courier_subset(
        table.candidates,
        selected_task_ids=small_tasks,
        courier_limit=22,
        salt="robust_small_hash",
        per_task_candidate_limit=58,
    )

    cases = [
        SolverCase(
            case_id="robust_full_anchor",
            input_text=_render_candidates(table.candidates),
            source_name=source_name,
        ),
        SolverCase(
            case_id="robust_lowwill_20_score_sensitive",
            input_text=_render_candidates(
                _transform_candidates(
                    _filter_candidates_by_tasks(table.candidates, low_tasks),
                    willingness_factor=0.72,
                    score_factor=1.25,
                )
            ),
            source_name=source_name,
            weight=2.0,
        ),
        SolverCase(
            case_id="robust_lowwill_full_soft",
            input_text=_render_candidates(
                _transform_candidates(
                    table.candidates,
                    willingness_factor=0.82,
                    score_factor=1.15,
                )
            ),
            source_name=source_name,
            weight=0.5,
        ),
        SolverCase(
            case_id="robust_large_lowwill_tail_soft",
            input_text=_render_candidates(
                _transform_low_willingness_tail(
                    table.candidates,
                    low_task_ids=tail_tasks,
                    low_factor=0.70,
                    other_factor=0.94,
                    low_score_factor=1.04,
                )
            ),
            source_name=source_name,
            weight=0.1,
        ),
        SolverCase(
            case_id="robust_medium_hash_25x50",
            input_text=_render_candidates(medium_candidates),
            source_name=source_name,
            weight=0.6,
        ),
        SolverCase(
            case_id="robust_medium_sparse_25x38",
            input_text=_render_candidates(medium_sparse_candidates),
            source_name=source_name,
            weight=1.2,
        ),
        SolverCase(
            case_id="robust_lowwill_sparse_20x34",
            input_text=_render_candidates(lowwill_candidates),
            source_name=source_name,
            weight=0.1,
        ),
        SolverCase(
            case_id="robust_scarce_soft_40x30",
            input_text=_render_candidates(scarce_soft_candidates),
            source_name=source_name,
            weight=0.05,
        ),
        SolverCase(
            case_id="robust_scarce_neutral_40x26",
            input_text=_render_candidates(scarce_neutral_candidates),
            source_name=source_name,
            weight=0.1,
        ),
        SolverCase(
            case_id="robust_small_hash_10x22",
            input_text=_render_candidates(small_candidates),
            source_name=source_name,
            weight=0.1,
        ),
    ]
    return [case for case in cases if _has_candidate_rows(case.input_text)]


def _filter_candidates_by_tasks(
    candidates: tuple[Candidate, ...],
    selected_task_ids: list[str],
) -> tuple[Candidate, ...]:
    """Keep only candidates whose full task list is inside the selected task set."""
    selected = set(selected_task_ids)
    return tuple(
        candidate
        for candidate in candidates
        if all(task_id in selected for task_id in candidate.task_id_list)
    )


def _filter_candidates_by_couriers(
    candidates: tuple[Candidate, ...],
    selected_courier_ids: list[str],
) -> tuple[Candidate, ...]:
    """Keep only candidates whose courier is inside the selected courier set."""
    selected = set(selected_courier_ids)
    return tuple(
        candidate for candidate in candidates if candidate.courier_id in selected
    )


def _build_task_courier_subset(
    candidates: tuple[Candidate, ...],
    *,
    selected_task_ids: list[str],
    courier_limit: int,
    salt: str,
    keep_ratio: float | None = None,
    min_candidates_per_task: int = 1,
    per_task_candidate_limit: int | None = None,
    score_jitter: float = 0.0,
    willingness_jitter: float = 0.0,
) -> tuple[Candidate, ...]:
    """Build a deterministic task/courier subset with optional sparsification."""
    task_filtered = _filter_candidates_by_tasks(candidates, selected_task_ids)
    selected_couriers = _select_coverage_balanced_couriers(
        task_filtered,
        selected_task_ids=selected_task_ids,
        limit=courier_limit,
        salt=salt,
    )
    subset = _filter_candidates_by_couriers(task_filtered, selected_couriers)
    if keep_ratio is not None:
        subset = _thin_candidates(
            subset,
            keep_ratio=keep_ratio,
            min_candidates_per_task=min_candidates_per_task,
            salt=salt,
        )
    if per_task_candidate_limit is not None:
        subset = _cap_candidates_per_task(
            subset,
            per_task_limit=per_task_candidate_limit,
            salt=salt,
        )
    if score_jitter or willingness_jitter:
        subset = _jitter_candidates(
            subset,
            score_range=score_jitter,
            willingness_range=willingness_jitter,
            salt=salt,
        )
    return subset


def _select_hash_ranked_items(
    items: list[str],
    *,
    limit: int,
    salt: str,
) -> list[str]:
    """Select a stable hash-ranked sample from a list of ids."""
    ranked = sorted(items, key=lambda item: (_stable_unit(salt, item), item))
    return ranked[:limit]


def _select_coverage_balanced_couriers(
    candidates: tuple[Candidate, ...],
    *,
    selected_task_ids: list[str],
    limit: int,
    salt: str,
) -> list[str]:
    """Select couriers that cover tasks without simply choosing busiest couriers."""
    selected_tasks = set(selected_task_ids)
    coverage_by_courier: dict[str, set[str]] = {}
    candidate_counts: dict[str, int] = {}
    for candidate in candidates:
        covered_tasks = set(candidate.task_id_list) & selected_tasks
        if not covered_tasks:
            continue
        courier_id = candidate.courier_id
        coverage_by_courier.setdefault(courier_id, set()).update(covered_tasks)
        candidate_counts[courier_id] = candidate_counts.get(courier_id, 0) + 1

    courier_ids = sorted(coverage_by_courier)
    if len(courier_ids) <= limit:
        return courier_ids

    sorted_counts = sorted(candidate_counts[courier_id] for courier_id in courier_ids)
    median_count = sorted_counts[len(sorted_counts) // 2]
    task_cover_counts = {task_id: 0 for task_id in selected_task_ids}
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < limit:
        best_courier = ""
        best_score: tuple[float, float, float, float] | None = None
        for courier_id in courier_ids:
            if courier_id in selected_set:
                continue
            covered_tasks = coverage_by_courier[courier_id]
            new_cover_count = sum(
                1 for task_id in covered_tasks if task_cover_counts[task_id] == 0
            )
            undercovered_score = sum(
                1.0 / (1.0 + task_cover_counts[task_id])
                for task_id in covered_tasks
            )
            balance_penalty = abs(candidate_counts[courier_id] - median_count)
            score = (
                float(new_cover_count),
                undercovered_score,
                -float(balance_penalty),
                -_stable_unit(salt, "courier", courier_id),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_courier = courier_id
        if not best_courier:
            break
        selected.append(best_courier)
        selected_set.add(best_courier)
        for task_id in coverage_by_courier[best_courier]:
            task_cover_counts[task_id] += 1
    return selected


def _thin_candidates(
    candidates: tuple[Candidate, ...],
    *,
    keep_ratio: float,
    min_candidates_per_task: int,
    salt: str,
) -> tuple[Candidate, ...]:
    """Deterministically drop candidate rows while repairing task coverage."""
    keep_ratio = max(0.0, min(keep_ratio, 1.0))
    kept = tuple(
        candidate
        for candidate in candidates
        if _stable_unit(
            salt,
            "thin",
            candidate.task_id_list_str,
            candidate.courier_id,
        )
        <= keep_ratio
    )
    return _repair_min_candidates_per_task(
        kept,
        source_candidates=candidates,
        min_candidates_per_task=min_candidates_per_task,
        salt=salt,
    )


def _repair_min_candidates_per_task(
    selected_candidates: tuple[Candidate, ...],
    *,
    source_candidates: tuple[Candidate, ...],
    min_candidates_per_task: int,
    salt: str,
) -> tuple[Candidate, ...]:
    """Add back high-quality rows until every task has enough candidates."""
    if min_candidates_per_task <= 0:
        return selected_candidates
    selected = list(selected_candidates)
    selected_keys = {
        (candidate.task_id_list_str, candidate.courier_id) for candidate in selected
    }
    counts = _candidate_counts_by_task(selected)
    task_ids = sorted({task_id for candidate in source_candidates for task_id in candidate.task_id_list})
    ranked_source = sorted(
        source_candidates,
        key=lambda candidate: _candidate_quality_key(candidate, salt=salt),
    )
    for task_id in task_ids:
        while counts.get(task_id, 0) < min_candidates_per_task:
            added = False
            for candidate in ranked_source:
                key = (candidate.task_id_list_str, candidate.courier_id)
                if key in selected_keys or task_id not in candidate.task_id_list:
                    continue
                selected.append(candidate)
                selected_keys.add(key)
                for covered_task_id in candidate.task_id_list:
                    counts[covered_task_id] = counts.get(covered_task_id, 0) + 1
                added = True
                break
            if not added:
                break
    return tuple(selected)


def _cap_candidates_per_task(
    candidates: tuple[Candidate, ...],
    *,
    per_task_limit: int,
    salt: str,
) -> tuple[Candidate, ...]:
    """Keep the best deterministic rows while limiting per-task density."""
    if per_task_limit <= 0:
        return candidates
    selected: list[Candidate] = []
    counts: dict[str, int] = {}
    for candidate in sorted(
        candidates,
        key=lambda candidate: _candidate_quality_key(candidate, salt=salt),
    ):
        if all(counts.get(task_id, 0) < per_task_limit for task_id in candidate.task_id_list):
            selected.append(candidate)
            for task_id in candidate.task_id_list:
                counts[task_id] = counts.get(task_id, 0) + 1
    return tuple(selected)


def _jitter_candidates(
    candidates: tuple[Candidate, ...],
    *,
    score_range: float,
    willingness_range: float,
    salt: str,
) -> tuple[Candidate, ...]:
    """Apply stable score and willingness perturbations."""
    jittered = []
    for candidate in candidates:
        score_delta = (
            2.0
            * _stable_unit(
                salt,
                "score",
                candidate.task_id_list_str,
                candidate.courier_id,
            )
            - 1.0
        )
        willingness_delta = (
            2.0
            * _stable_unit(
                salt,
                "willingness",
                candidate.task_id_list_str,
                candidate.courier_id,
            )
            - 1.0
        )
        jittered.append(
            Candidate(
                task_id_list=candidate.task_id_list,
                task_id_list_str=candidate.task_id_list_str,
                courier_id=candidate.courier_id,
                total_score=candidate.total_score * (1.0 + score_delta * score_range),
                willingness=max(
                    0.0,
                    min(candidate.willingness + willingness_delta * willingness_range, 1.0),
                ),
            )
        )
    return tuple(jittered)


def _transform_low_willingness_tail(
    candidates: tuple[Candidate, ...],
    *,
    low_task_ids: list[str],
    low_factor: float,
    other_factor: float,
    low_score_factor: float,
) -> tuple[Candidate, ...]:
    """Stress low-willingness tails without changing candidate topology."""
    low_tasks = set(low_task_ids)
    transformed = []
    for candidate in candidates:
        touches_low_task = any(task_id in low_tasks for task_id in candidate.task_id_list)
        willingness_factor = low_factor if touches_low_task else other_factor
        score_factor = low_score_factor if touches_low_task else 1.0
        transformed.append(
            Candidate(
                task_id_list=candidate.task_id_list,
                task_id_list_str=candidate.task_id_list_str,
                courier_id=candidate.courier_id,
                total_score=candidate.total_score * score_factor,
                willingness=max(0.0, min(candidate.willingness * willingness_factor, 1.0)),
            )
        )
    return tuple(transformed)


def _candidate_counts_by_task(candidates: list[Candidate]) -> dict[str, int]:
    """Return candidate-row counts for each covered task."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        for task_id in candidate.task_id_list:
            counts[task_id] = counts.get(task_id, 0) + 1
    return counts


def _candidate_quality_key(candidate: Candidate, *, salt: str) -> tuple[float, float, float]:
    """Rank candidate rows by cheap risk-aware quality."""
    task_size = float(len(candidate.task_id_list) or 1)
    return (
        candidate.total_score / task_size - 90.0 * candidate.willingness,
        candidate.total_score,
        _stable_unit(salt, "quality", candidate.task_id_list_str, candidate.courier_id),
    )


def _stable_unit(*parts: object) -> float:
    """Return a deterministic pseudo-random unit interval value."""
    raw = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _transform_candidates(
    candidates: tuple[Candidate, ...],
    *,
    willingness_factor: float = 1.0,
    willingness_offset: float = 0.0,
    score_factor: float = 1.0,
    bundle_score_factor: float = 1.0,
    bundle_willingness_offset: float = 0.0,
) -> tuple[Candidate, ...]:
    """Return candidates with deterministic score/willingness transforms."""
    transformed = []
    for candidate in candidates:
        is_bundle = len(candidate.task_id_list) > 1
        transformed.append(
            Candidate(
                task_id_list=candidate.task_id_list,
                task_id_list_str=candidate.task_id_list_str,
                courier_id=candidate.courier_id,
                total_score=(
                    candidate.total_score
                    * score_factor
                    * (bundle_score_factor if is_bundle else 1.0)
                ),
                willingness=max(
                    0.0,
                    min(
                        candidate.willingness * willingness_factor
                        + willingness_offset
                        + (bundle_willingness_offset if is_bundle else 0.0),
                        1.0,
                    ),
                ),
            )
        )
    return tuple(transformed)


def _select_low_willingness_tasks(
    candidates: tuple[Candidate, ...],
    *,
    limit: int,
) -> list[str]:
    """Select tasks with the lowest average candidate willingness."""
    willingness_by_task: dict[str, list[float]] = {}
    for candidate in candidates:
        for task_id in candidate.task_id_list:
            willingness_by_task.setdefault(task_id, []).append(candidate.willingness)
    ranked = sorted(
        willingness_by_task,
        key=lambda task_id: (
            sum(willingness_by_task[task_id]) / len(willingness_by_task[task_id]),
            task_id,
        ),
    )
    return ranked[:limit]


def _render_candidates(candidates: tuple[Candidate, ...]) -> str:
    """Render candidate rows back to the reference tab-separated text shape."""
    lines = ["task_id_list\tcourier_id\ttotal_score\twillingness"]
    for candidate in candidates:
        lines.append(
            "\t".join(
                [
                    candidate.task_id_list_str,
                    candidate.courier_id,
                    str(candidate.total_score),
                    str(candidate.willingness),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _has_candidate_rows(input_text: str) -> bool:
    """Return whether rendered input contains at least one candidate row."""
    return len(input_text.strip().splitlines()) > 1
