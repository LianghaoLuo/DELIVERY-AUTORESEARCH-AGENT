"""Primitive-based standalone solver rendering."""

from __future__ import annotations

import json
from typing import Any


def render_solver_source(config_payload: dict[str, Any]) -> str:
    """Render one StrategyConfig payload into a Python 3.6 solver source."""
    header = json.dumps(config_payload, sort_keys=True)
    source = _SOLVER_TEMPLATE.replace("__STRATEGY_CONFIG_JSON__", repr(header))
    return f"# StrategyConfig: {header}\n{source}"


_SOLVER_TEMPLATE = r'''"""Generated primitive strategy solver.

This file is standalone and Python 3.6 compatible.
"""

import json
import time


STRATEGY_CONFIG = json.loads(__STRATEGY_CONFIG_JSON__)


def _parse_candidates(input_text):
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    candidates = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        task_ids = tuple(
            task.strip() for task in task_id_list_str.split(",") if task.strip()
        )
        courier_id = courier_id.strip()
        if not task_ids or not courier_id:
            continue
        normalized_task_list = ",".join(task_ids)
        candidates.append(
            {
                "task_id_list_str": normalized_task_list,
                "task_ids": task_ids,
                "courier_id": courier_id,
                "score": score,
                "willingness": willingness,
            }
        )
    return candidates


def _clamp_probability(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _combined_success_probability(probabilities):
    failure = 1.0
    for probability in probabilities:
        failure *= 1.0 - _clamp_probability(probability)
    return 1.0 - failure


def _all_task_ids(candidates):
    task_ids = set()
    for candidate in candidates:
        for task_id in candidate["task_ids"]:
            task_ids.add(task_id)
    return task_ids


def _candidate_groups(candidates):
    by_task_list = {}
    by_task = {}
    option_counts = {}
    for candidate in candidates:
        by_task_list.setdefault(candidate["task_id_list_str"], []).append(candidate)
        for task_id in candidate["task_ids"]:
            by_task.setdefault(task_id, []).append(candidate)
            option_counts[task_id] = option_counts.get(task_id, 0) + 1
    return by_task_list, by_task, option_counts


def _merged_params(config, step=None):
    params = {}
    primary = config.get("primary", {})
    params.update(primary.get("params", {}) or {})
    control = config.get("control", {})
    params.update(control.get("params", {}) or {})
    if step is not None:
        params.update(step.get("params", {}) or {})
    return params


def _control(config, name, default):
    control = config.get("control", {})
    value = control.get(name)
    return default if value is None else value


def _candidate_objective(candidate, primary_kind, params, option_counts):
    score = float(candidate["score"])
    willingness = _clamp_probability(float(candidate["willingness"]))
    bundle_size = len(candidate["task_ids"])
    alpha = float(params.get("alpha", 90.0))
    score_weight = float(params.get("score_weight", 1.0))
    failure_weight = float(
        params.get("failure_weight", params.get("expected_failed_weight", 0.0))
    )
    success_credit = float(
        params.get("success_credit", params.get("expected_success_credit", alpha))
    )
    bundle_bias = float(params.get("bundle_bias", 0.0))
    if primary_kind == "greedy":
        return score
    if primary_kind == "bundle_merge":
        bonus = float(params.get("scarce_bundle_bonus", 40.0))
        return score - alpha * willingness - (bonus if bundle_size > 1 else 0.0)
    if primary_kind == "bundle_split":
        penalty = abs(float(params.get("split_min_improvement", 10.0)))
        return score - alpha * willingness + (penalty if bundle_size > 1 else 0.0)
    if primary_kind == "beam":
        pair_penalty = float(params.get("beam_pair_penalty", 2.0))
        bundle_penalty = float(params.get("beam_bundle_penalty", 0.0))
        return (
            score
            - float(params.get("beam_success_credit", success_credit)) * willingness
            + pair_penalty * max(bundle_size - 1, 0)
            + (bundle_penalty if bundle_size > 1 else 0.0)
        )
    if primary_kind == "regret_rank":
        scarcity = 0.0
        for task_id in candidate["task_ids"]:
            scarcity += 1.0 / float(max(option_counts.get(task_id, 1), 1))
        return (
            float(params.get("regret_score_weight", score_weight)) * score
            - float(params.get("regret_willingness_weight", 35.0)) * willingness
            - float(params.get("regret_scarcity_weight", 80.0)) * scarcity
            - (float(params.get("regret_bundle_bonus", 0.0)) if bundle_size > 1 else 0.0)
        )
    return (
        score_weight * score
        + failure_weight * (1.0 - willingness)
        - success_credit * willingness
        + (bundle_bias if bundle_size > 1 else 0.0)
    )


def _ranked_candidates(candidates, primary_kind, params, option_counts):
    if primary_kind == "bundle_merge":
        prefix = lambda candidate: 0 if len(candidate["task_ids"]) > 1 else 1
    elif primary_kind == "bundle_split":
        prefix = lambda candidate: 0 if len(candidate["task_ids"]) == 1 else 1
    else:
        prefix = lambda candidate: 0
    return sorted(
        candidates,
        key=lambda candidate: (
            prefix(candidate),
            _candidate_objective(candidate, primary_kind, params, option_counts),
            candidate["score"],
            candidate["task_id_list_str"],
            candidate["courier_id"],
        ),
    )


def _select_primary(candidates, config, started_at):
    primary = config.get("primary", {})
    primary_kind = primary.get("kind", "willingness_adjusted")
    params = _merged_params(config)
    by_task_list, by_task, option_counts = _candidate_groups(candidates)
    ranked = _ranked_candidates(candidates, primary_kind, params, option_counts)
    task_count = len(_all_task_ids(candidates))
    selected = []
    assigned_tasks = set()
    used_couriers = set()
    for candidate in ranked:
        if _time_exceeded(config, started_at, 0.45):
            break
        if candidate["courier_id"] in used_couriers:
            continue
        if any(task_id in assigned_tasks for task_id in candidate["task_ids"]):
            continue
        selected.append(candidate)
        used_couriers.add(candidate["courier_id"])
        for task_id in candidate["task_ids"]:
            assigned_tasks.add(task_id)
        if len(assigned_tasks) >= task_count:
            break

    for task_id in sorted(_all_task_ids(candidates) - assigned_tasks):
        if _time_exceeded(config, started_at, 0.55):
            break
        singles = [
            candidate
            for candidate in by_task.get(task_id, [])
            if len(candidate["task_ids"]) == 1
            and candidate["courier_id"] not in used_couriers
        ]
        if not singles:
            continue
        candidate = _ranked_candidates(singles, primary_kind, params, option_counts)[0]
        selected.append(candidate)
        used_couriers.add(candidate["courier_id"])
        assigned_tasks.add(task_id)
    return selected


def _time_exceeded(config, started_at, fraction):
    budget = _control(config, "time_budget_seconds", 8.5)
    return time.time() - started_at > float(budget) * fraction


def _local_improve(selected, candidates, config, step, started_at):
    primary = config.get("primary", {})
    primary_kind = primary.get("kind", "willingness_adjusted")
    params = _merged_params(config, step)
    _, _, option_counts = _candidate_groups(candidates)
    max_passes = int(params.get("max_passes", 2))
    selected = list(selected)
    for _ in range(max_passes):
        changed = False
        used_by_index = [candidate["courier_id"] for candidate in selected]
        for index, current in enumerate(list(selected)):
            if _time_exceeded(config, started_at, 0.72):
                return selected
            other_couriers = set(used_by_index)
            other_couriers.discard(current["courier_id"])
            current_score = _candidate_objective(
                current, primary_kind, params, option_counts
            )
            replacements = [
                candidate
                for candidate in candidates
                if candidate["task_id_list_str"] == current["task_id_list_str"]
                and candidate["courier_id"] not in other_couriers
            ]
            if not replacements:
                continue
            best = min(
                replacements,
                key=lambda candidate: (
                    _candidate_objective(candidate, primary_kind, params, option_counts),
                    candidate["score"],
                    candidate["courier_id"],
                ),
            )
            best_score = _candidate_objective(best, primary_kind, params, option_counts)
            if best_score + 1e-9 < current_score:
                selected[index] = best
                used_by_index[index] = best["courier_id"]
                changed = True
        if not changed:
            break
    return selected


def _assignments_from_candidates(selected):
    assignments = []
    for candidate in selected:
        assignments.append(
            {
                "task_id_list_str": candidate["task_id_list_str"],
                "task_ids": candidate["task_ids"],
                "courier_ids": [candidate["courier_id"]],
                "probabilities": [_clamp_probability(candidate["willingness"])],
                "total_score": float(candidate["score"]),
            }
        )
    return assignments


def _used_couriers(assignments):
    used = set()
    for assignment in assignments:
        for courier_id in assignment["courier_ids"]:
            used.add(courier_id)
    return used


def _assignment_success(assignment):
    return _combined_success_probability(assignment["probabilities"])


def _repair_target(assignment, config, step):
    params = _merged_params(config, step)
    current = _assignment_success(assignment)
    if current < float(params.get("mid_risk_target", _control(config, "mid_risk_target", 0.90))):
        return float(params.get("mid_risk_target", _control(config, "mid_risk_target", 0.90)))
    return float(params.get("high_risk_target", _control(config, "high_risk_target", 0.92)))


def _duplicate_repair(assignments, candidates, config, step, started_at, task_overlay):
    params = _merged_params(config, step)
    by_task_list, by_task, _ = _candidate_groups(candidates)
    used = _used_couriers(assignments)
    max_extra = int(params.get("max_extra_dispatches", _control(config, "max_extra_dispatches", 20)))
    max_couriers = int(
        params.get(
            "max_couriers_per_assignment",
            _control(config, "max_couriers_per_assignment", 3),
        )
    )
    min_success = float(params.get("min_success_probability", 0.0))
    min_roi = float(params.get("min_roi", _control(config, "min_roi", 0.0)))
    score_weight = float(params.get("score_weight", 1.0))
    success_credit = float(params.get("expected_success_credit", 100.0))
    expected_failed_weight = float(params.get("expected_failed_weight", 0.0))
    extra = 0
    while extra < max_extra:
        if _time_exceeded(config, started_at, 0.94):
            break
        best = None
        task_probabilities = _task_probabilities(assignments)
        for assignment in list(assignments):
            current_success = _assignment_success(assignment)
            target = max(min_success, _repair_target(assignment, config, step))
            if len(assignment["courier_ids"]) >= max_couriers:
                continue
            if current_success >= target and min_roi >= 0.0:
                continue
            pool = by_task_list.get(assignment["task_id_list_str"], [])
            for candidate in pool:
                if candidate["courier_id"] in used:
                    continue
                probability = _clamp_probability(candidate["willingness"])
                success_gain = (1.0 - current_success) * probability * len(assignment["task_ids"])
                raw_roi = (
                    (expected_failed_weight + success_credit) * success_gain
                    - score_weight * float(candidate["score"])
                )
                if raw_roi < min_roi:
                    continue
                rank = (
                    raw_roi,
                    success_gain,
                    probability,
                    -float(candidate["score"]),
                    candidate["courier_id"],
                )
                if best is None or rank > best[0]:
                    best = (rank, assignment, candidate, probability)
            if not task_overlay:
                continue
            for task_id in assignment["task_ids"]:
                current_task_success = _combined_success_probability(
                    task_probabilities.get(task_id, [])
                )
                if current_task_success >= target and min_roi >= 0.0:
                    continue
                for candidate in by_task.get(task_id, []):
                    if len(candidate["task_ids"]) != 1:
                        continue
                    if candidate["courier_id"] in used:
                        continue
                    probability = _clamp_probability(candidate["willingness"])
                    success_gain = (1.0 - current_task_success) * probability
                    raw_roi = (
                        (expected_failed_weight + success_credit) * success_gain
                        - score_weight * float(candidate["score"])
                    )
                    if raw_roi < min_roi:
                        continue
                    rank = (
                        raw_roi + 2.0,
                        success_gain,
                        probability,
                        -float(candidate["score"]),
                        candidate["courier_id"],
                    )
                    if best is None or rank > best[0]:
                        best = (rank, None, candidate, probability)
        if best is None:
            break
        _, assignment, candidate, probability = best
        if assignment is None:
            assignments.append(
                {
                    "task_id_list_str": candidate["task_id_list_str"],
                    "task_ids": candidate["task_ids"],
                    "courier_ids": [candidate["courier_id"]],
                    "probabilities": [probability],
                    "total_score": float(candidate["score"]),
                }
            )
        else:
            assignment["courier_ids"].append(candidate["courier_id"])
            assignment["probabilities"].append(probability)
            assignment["total_score"] += float(candidate["score"])
        used.add(candidate["courier_id"])
        extra += 1
    return assignments


def _task_probabilities(assignments):
    probabilities = {}
    for assignment in assignments:
        for probability in assignment["probabilities"]:
            for task_id in assignment["task_ids"]:
                probabilities.setdefault(task_id, []).append(probability)
    return probabilities


def _drop_tail_probe(assignments, config, step):
    params = _merged_params(config, step)
    max_drop = int(params.get("max_drop_tasks", 0))
    drop_ratio = float(params.get("drop_ratio", 0.0))
    min_task_count = int(params.get("min_task_count", 10))
    task_count = sum(len(assignment["task_ids"]) for assignment in assignments)
    if max_drop <= 0 or drop_ratio <= 0.0 or task_count < min_task_count:
        return assignments
    drop_budget = min(max_drop, int(task_count * drop_ratio))
    if drop_budget <= 0:
        return assignments
    ranked = sorted(
        assignments,
        key=lambda assignment: (
            _assignment_success(assignment),
            -float(assignment["total_score"]),
        ),
    )
    dropped = set()
    for assignment in ranked:
        if len(dropped) + len(assignment["task_ids"]) > drop_budget:
            continue
        dropped.add(assignment["task_id_list_str"])
    if not dropped:
        return assignments
    return [
        assignment
        for assignment in assignments
        if assignment["task_id_list_str"] not in dropped
    ]


def _format_output(assignments):
    return [
        (assignment["task_id_list_str"], list(assignment["courier_ids"]))
        for assignment in assignments
        if assignment["courier_ids"]
    ]


def solve(input_text):
    candidates = _parse_candidates(input_text)
    started_at = time.time()
    selected = _select_primary(candidates, STRATEGY_CONFIG, started_at)
    for step in STRATEGY_CONFIG.get("repairs", []):
        if step.get("kind") == "local_improve":
            selected = _local_improve(
                selected, candidates, STRATEGY_CONFIG, step, started_at
            )
    assignments = _assignments_from_candidates(selected)
    for step in STRATEGY_CONFIG.get("repairs", []):
        kind = step.get("kind")
        if kind in ("duplicate_dispatch", "risk_tier_duplicate", "staged_duplicate"):
            assignments = _duplicate_repair(
                assignments, candidates, STRATEGY_CONFIG, step, started_at, False
            )
        elif kind == "task_overlay":
            assignments = _duplicate_repair(
                assignments, candidates, STRATEGY_CONFIG, step, started_at, True
            )
        elif kind == "drop_tail_probe":
            assignments = _drop_tail_probe(assignments, STRATEGY_CONFIG, step)
    return _format_output(assignments)
'''
