"""Current local-best prior solver.

Params: {'alpha': 90.0, 'beam_width': 14, 'bundle_bonus': 26.0, 'duplicate_high_target': 0.97, 'duplicate_max_couriers': 3, 'duplicate_max_extra': 45, 'duplicate_min_gain': -2.0, 'duplicate_pair_penalty': 8.0, 'duplicate_target': 0.94, 'expected_failed_weight': 150.0, 'expected_success_credit': 430.0, 'low_willingness_best_threshold': 0.9, 'max_repair_rounds': 56, 'neighborhood_size': 10, 'pair_penalty': 6.0, 'pressure_weight': 58.0, 'profile': 'repair_deep', 'scarce_duplicate_cap': 3, 'scarce_ratio_threshold': 1.35, 'score_weight': 1.0, 'shadow_weight': 0.2, 'time_budget_seconds': 8.5, 'top_candidates_per_task': 10}
Python 3.6-compatible candidate solver. Do not import agent-side modules here.
"""

import time

PROFILE = "repair_deep"
ALPHA = 90.0
SCORE_WEIGHT = 1.0
EXPECTED_FAILED_WEIGHT = 150.0
EXPECTED_SUCCESS_CREDIT = 430.0
UNASSIGNED_TASK_WEIGHT = 10000.0
PAIR_PENALTY = 6.0
DUPLICATE_PAIR_PENALTY = 8.0
SHADOW_WEIGHT = 0.2
PRESSURE_WEIGHT = 58.0
BUNDLE_BONUS = 26.0
SCARCE_RATIO_THRESHOLD = 1.35
LOW_WILLINGNESS_BEST_THRESHOLD = 0.9
TIME_BUDGET_SECONDS = 8.5
MAX_REPAIR_ROUNDS = 56
NEIGHBORHOOD_SIZE = 10
BEAM_WIDTH = 14
TOP_CANDIDATES_PER_TASK = 10
DUPLICATE_TARGET = 0.94
DUPLICATE_HIGH_TARGET = 0.97
DUPLICATE_MAX_EXTRA = 45
DUPLICATE_MAX_COURIERS = 3
DUPLICATE_MIN_GAIN = -2.0
SCARCE_DUPLICATE_CAP = 3


def _parse_candidates(input_text):
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    candidates = []
    index = 0
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
        task_ids = [task.strip() for task in task_id_list_str.split(",") if task.strip()]
        courier_id = courier_id.strip()
        if not task_ids or not courier_id:
            continue
        candidates.append({
            "index": index,
            "score": score,
            "task_id_list_str": ",".join(task_ids),
            "task_ids": task_ids,
            "courier_id": courier_id,
            "willingness": willingness,
        })
        index += 1
    return candidates


def solve(input_text):
    """Return a global-risk repaired assignment with bounded LNS."""
    started_at = time.time()
    candidates = _parse_candidates(input_text)
    if not candidates:
        return []
    context = _build_context(candidates)
    selected = _initial_solution(context, started_at)
    selected = _repair_uncovered_tasks(selected, context, started_at)
    selected = _lns_repair(selected, context, started_at)
    assignments = _to_assignments(selected)
    assignments = _augment_duplicates(assignments, context, started_at)
    return [
        (assignment["task_id_list_str"], list(assignment["courier_ids"]))
        for assignment in assignments
        if assignment["courier_ids"]
    ]


def _build_context(candidates):
    tasks = sorted(set(task_id for candidate in candidates for task_id in candidate["task_ids"]))
    couriers = sorted(set(candidate["courier_id"] for candidate in candidates))
    by_task = {}
    by_task_list = {}
    for candidate in candidates:
        by_task_list.setdefault(candidate["task_id_list_str"], []).append(candidate)
        for task_id in candidate["task_ids"]:
            by_task.setdefault(task_id, []).append(candidate)
    task_pressure = _task_pressure(tasks, by_task)
    courier_shadow = _courier_shadow(candidates, task_pressure)
    task_count = len(tasks)
    courier_count = len(couriers)
    scarce_mode = (
        task_count > 0
        and float(courier_count) / float(task_count) <= SCARCE_RATIO_THRESHOLD
    )
    low_input_mode = _is_low_willingness_input(tasks, by_task)
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: _candidate_sort_key(
            candidate,
            task_pressure,
            courier_shadow,
            scarce_mode,
            low_input_mode,
        ),
    )
    ranked_by_task = {}
    for task_id, task_candidates in by_task.items():
        ranked_by_task[task_id] = sorted(
            task_candidates,
            key=lambda candidate: _candidate_sort_key(
                candidate,
                task_pressure,
                courier_shadow,
                scarce_mode,
                low_input_mode,
            ),
        )[: max(TOP_CANDIDATES_PER_TASK * 3, TOP_CANDIDATES_PER_TASK)]
    return {
        "candidates": candidates,
        "ranked_candidates": ranked_candidates,
        "ranked_by_task": ranked_by_task,
        "by_task_list": by_task_list,
        "tasks": tasks,
        "couriers": couriers,
        "task_pressure": task_pressure,
        "courier_shadow": courier_shadow,
        "task_count": task_count,
        "courier_count": courier_count,
        "scarce_mode": scarce_mode,
        "low_input_mode": low_input_mode,
    }


def _task_pressure(tasks, by_task):
    pressure = {}
    for task_id in tasks:
        candidates = by_task.get(task_id, [])
        best_willingness = 0.0
        objectives = []
        for candidate in candidates:
            task_size = float(len(candidate["task_ids"]) or 1)
            willingness = _clamp_probability(candidate["willingness"])
            if willingness > best_willingness:
                best_willingness = willingness
            objectives.append(candidate["score"] / task_size - ALPHA * willingness)
        objectives.sort()
        if len(objectives) >= 2:
            regret = max(objectives[1] - objectives[0], 0.0)
        else:
            regret = 60.0
        count = float(max(len(candidates), 1))
        count_pressure = min(1.0, 18.0 / count)
        willingness_pressure = max(1.0 - best_willingness, 0.0)
        regret_pressure = min(regret / 80.0, 1.0)
        pressure[task_id] = (
            0.50 * willingness_pressure
            + 0.30 * regret_pressure
            + 0.20 * count_pressure
        )
    return pressure


def _courier_shadow(candidates, task_pressure):
    raw = {}
    for candidate in candidates:
        pressure_sum = sum(task_pressure.get(task_id, 0.0) for task_id in candidate["task_ids"])
        raw[candidate["courier_id"]] = raw.get(candidate["courier_id"], 0.0) + pressure_sum
    max_raw = max(raw.values()) if raw else 0.0
    if max_raw <= 0.0:
        return {courier_id: 0.0 for courier_id in raw}
    return {courier_id: 100.0 * value / max_raw for courier_id, value in raw.items()}


def _is_low_willingness_input(tasks, by_task):
    if not tasks:
        return False
    low_count = 0
    total_best = 0.0
    for task_id in tasks:
        best = 0.0
        for candidate in by_task.get(task_id, []):
            best = max(best, _clamp_probability(candidate["willingness"]))
        total_best += best
        if best < LOW_WILLINGNESS_BEST_THRESHOLD:
            low_count += 1
    low_ratio = float(low_count) / float(len(tasks))
    average_best = total_best / float(len(tasks))
    return low_ratio >= 0.25 or average_best < LOW_WILLINGNESS_BEST_THRESHOLD


def _candidate_sort_key(candidate, task_pressure, courier_shadow, scarce_mode, low_input_mode):
    task_size = float(len(candidate["task_ids"]) or 1)
    pressure_sum = sum(task_pressure.get(task_id, 0.0) for task_id in candidate["task_ids"])
    willingness = _clamp_probability(candidate["willingness"])
    bundle_bonus = BUNDLE_BONUS if len(candidate["task_ids"]) > 1 else 0.0
    if scarce_mode and len(candidate["task_ids"]) > 1:
        bundle_bonus += 0.75 * BUNDLE_BONUS
    low_bonus = 10.0 if low_input_mode and len(candidate["task_ids"]) == 1 else 0.0
    shadow = courier_shadow.get(candidate["courier_id"], 0.0)
    return (
        candidate["score"] / task_size
        - ALPHA * willingness
        - PRESSURE_WEIGHT * pressure_sum / task_size
        - bundle_bonus
        - low_bonus
        + SHADOW_WEIGHT * shadow,
        candidate["score"],
        candidate["index"],
    )


def _initial_solution(context, started_at):
    selected = []
    used_couriers = set()
    covered_tasks = set()
    for candidate in context["ranked_candidates"]:
        if time.time() - started_at > TIME_BUDGET_SECONDS * 0.18:
            break
        if candidate["courier_id"] in used_couriers:
            continue
        if any(task_id in covered_tasks for task_id in candidate["task_ids"]):
            continue
        selected.append(candidate)
        used_couriers.add(candidate["courier_id"])
        for task_id in candidate["task_ids"]:
            covered_tasks.add(task_id)
        if len(covered_tasks) >= context["task_count"]:
            break
    return selected


def _repair_uncovered_tasks(selected, context, started_at):
    selected = list(selected)
    used_couriers = set(candidate["courier_id"] for candidate in selected)
    covered_tasks = set(task_id for candidate in selected for task_id in candidate["task_ids"])
    while len(covered_tasks) < context["task_count"]:
        if time.time() - started_at > TIME_BUDGET_SECONDS * 0.28:
            break
        missing = [task_id for task_id in context["tasks"] if task_id not in covered_tasks]
        best = None
        for task_id in missing:
            for candidate in context["ranked_by_task"].get(task_id, []):
                if candidate["courier_id"] in used_couriers:
                    continue
                if any(covered_task in covered_tasks for covered_task in candidate["task_ids"]):
                    continue
                pressure = sum(context["task_pressure"].get(item, 0.0) for item in candidate["task_ids"])
                rank_key = (
                    len(candidate["task_ids"]),
                    pressure,
                    _clamp_probability(candidate["willingness"]),
                    -candidate["score"],
                )
                if best is None or rank_key > best[0]:
                    best = (rank_key, candidate)
        if best is None:
            break
        _, candidate = best
        selected.append(candidate)
        used_couriers.add(candidate["courier_id"])
        for task_id in candidate["task_ids"]:
            covered_tasks.add(task_id)
    return selected


def _lns_repair(selected, context, started_at):
    selected = list(selected)
    current_objective = _primary_objective(selected, context)
    ranked_tasks = _rank_pressure_tasks(context)
    if not ranked_tasks:
        return selected
    for round_index in range(MAX_REPAIR_ROUNDS):
        if time.time() - started_at > TIME_BUDGET_SECONDS * 0.82:
            break
        start = (round_index * max(1, NEIGHBORHOOD_SIZE // 2)) % len(ranked_tasks)
        neighborhood = []
        for offset in range(min(NEIGHBORHOOD_SIZE, len(ranked_tasks))):
            neighborhood.append(ranked_tasks[(start + offset) % len(ranked_tasks)])
        neighborhood_set = set(neighborhood)
        kept = []
        released_tasks = set()
        for candidate in selected:
            if any(task_id in neighborhood_set for task_id in candidate["task_ids"]):
                for task_id in candidate["task_ids"]:
                    released_tasks.add(task_id)
            else:
                kept.append(candidate)
        if not released_tasks:
            continue
        patch = _beam_rebuild_patch(kept, released_tasks, context, started_at)
        trial = kept + patch
        trial = _repair_uncovered_tasks(trial, context, started_at)
        trial_objective = _primary_objective(trial, context)
        if trial_objective + 1e-9 < current_objective:
            selected = trial
            current_objective = trial_objective
    return selected


def _rank_pressure_tasks(context):
    selected_pressure = context["task_pressure"]
    return sorted(
        context["tasks"],
        key=lambda task_id: (selected_pressure.get(task_id, 0.0), task_id),
        reverse=True,
    )


def _beam_rebuild_patch(kept, released_tasks, context, started_at):
    occupied_tasks = set(task_id for candidate in kept for task_id in candidate["task_ids"])
    occupied_couriers = set(candidate["courier_id"] for candidate in kept)
    missing_tasks = set(task_id for task_id in released_tasks if task_id not in occupied_tasks)
    pool = _neighborhood_pool(missing_tasks, occupied_tasks, occupied_couriers, context)
    states = [{"selected": [], "tasks": set(), "couriers": set()}]
    for candidate in pool:
        if time.time() - started_at > TIME_BUDGET_SECONDS * 0.78:
            break
        expanded = list(states)
        for state in states:
            if candidate["courier_id"] in state["couriers"]:
                continue
            if any(task_id in state["tasks"] for task_id in candidate["task_ids"]):
                continue
            if not any(task_id in missing_tasks for task_id in candidate["task_ids"]):
                continue
            new_tasks = set(state["tasks"])
            for task_id in candidate["task_ids"]:
                new_tasks.add(task_id)
            new_couriers = set(state["couriers"])
            new_couriers.add(candidate["courier_id"])
            expanded.append({
                "selected": state["selected"] + [candidate],
                "tasks": new_tasks,
                "couriers": new_couriers,
            })
        states = _prune_patch_states(expanded, kept, context)
    best_state = min(states, key=lambda state: _primary_objective(kept + state["selected"], context))
    return best_state["selected"]


def _neighborhood_pool(missing_tasks, occupied_tasks, occupied_couriers, context):
    pool = []
    seen = set()
    for task_id in sorted(missing_tasks):
        count = 0
        for candidate in context["ranked_by_task"].get(task_id, []):
            if candidate["courier_id"] in occupied_couriers:
                continue
            if any(candidate_task in occupied_tasks for candidate_task in candidate["task_ids"]):
                continue
            if not all(candidate_task in missing_tasks for candidate_task in candidate["task_ids"]):
                continue
            key = (candidate["task_id_list_str"], candidate["courier_id"])
            if key in seen:
                continue
            seen.add(key)
            pool.append(candidate)
            count += 1
            if count >= TOP_CANDIDATES_PER_TASK:
                break
    pool.sort(key=lambda candidate: _candidate_sort_key(
        candidate,
        context["task_pressure"],
        context["courier_shadow"],
        context["scarce_mode"],
        context["low_input_mode"],
    ))
    return pool


def _prune_patch_states(states, kept, context):
    states.sort(key=lambda state: _primary_objective(kept + state["selected"], context))
    pruned = []
    seen = set()
    for state in states:
        signature = (
            tuple(sorted(state["tasks"])),
            tuple(sorted(state["couriers"])),
        )
        if signature in seen:
            continue
        pruned.append(state)
        seen.add(signature)
        if len(pruned) >= BEAM_WIDTH:
            break
    return pruned


def _primary_objective(selected, context):
    total_score = 0.0
    probabilities_by_task = {}
    pair_count = 0
    bundle_covered_count = 0
    used_shadow = 0.0
    for candidate in selected:
        total_score += candidate["score"]
        pair_count += 1
        used_shadow += context["courier_shadow"].get(candidate["courier_id"], 0.0)
        probability = _clamp_probability(candidate["willingness"])
        if len(candidate["task_ids"]) > 1:
            bundle_covered_count += len(candidate["task_ids"])
        for task_id in candidate["task_ids"]:
            probabilities_by_task.setdefault(task_id, []).append(probability)
    expected_success = sum(_combined_success_probability(values) for values in probabilities_by_task.values())
    unassigned = max(context["task_count"] - len(probabilities_by_task), 0)
    expected_failed = max(context["task_count"] - expected_success, 0.0)
    return (
        SCORE_WEIGHT * total_score
        + EXPECTED_FAILED_WEIGHT * expected_failed
        - EXPECTED_SUCCESS_CREDIT * expected_success
        + UNASSIGNED_TASK_WEIGHT * unassigned
        + PAIR_PENALTY * pair_count
        + SHADOW_WEIGHT * used_shadow
        - BUNDLE_BONUS * 0.25 * bundle_covered_count
    )


def _to_assignments(selected):
    assignments = []
    for candidate in selected:
        assignments.append({
            "task_id_list_str": candidate["task_id_list_str"],
            "task_ids": candidate["task_ids"],
            "courier_ids": [candidate["courier_id"]],
            "probabilities": [_clamp_probability(candidate["willingness"])],
            "total_score": candidate["score"],
        })
    return assignments


def _augment_duplicates(assignments, context, started_at):
    used_couriers = set()
    for assignment in assignments:
        for courier_id in assignment["courier_ids"]:
            used_couriers.add(courier_id)
    max_extra = DUPLICATE_MAX_EXTRA
    if context["scarce_mode"]:
        max_extra = min(max_extra, SCARCE_DUPLICATE_CAP)
    extra_count = 0
    while extra_count < max_extra:
        if time.time() - started_at > TIME_BUDGET_SECONDS:
            break
        best = None
        for assignment in assignments:
            if len(assignment["courier_ids"]) >= DUPLICATE_MAX_COURIERS:
                continue
            current_success = _combined_success_probability(assignment["probabilities"])
            target = _duplicate_target(current_success, context)
            if current_success >= target:
                continue
            for candidate in context["by_task_list"].get(assignment["task_id_list_str"], []):
                if candidate["courier_id"] in used_couriers:
                    continue
                probability = _clamp_probability(candidate["willingness"])
                success_gain = (1.0 - current_success) * probability * len(assignment["task_ids"])
                shadow = context["courier_shadow"].get(candidate["courier_id"], 0.0)
                gain = (
                    (EXPECTED_FAILED_WEIGHT + EXPECTED_SUCCESS_CREDIT) * success_gain
                    - SCORE_WEIGHT * candidate["score"]
                    - DUPLICATE_PAIR_PENALTY
                    - SHADOW_WEIGHT * shadow
                )
                if gain < DUPLICATE_MIN_GAIN:
                    continue
                risk_gap = max(target - current_success, 0.0) * len(assignment["task_ids"])
                rank_key = (
                    risk_gap,
                    gain,
                    success_gain,
                    probability,
                    -candidate["score"],
                )
                if best is None or rank_key > best[0]:
                    best = (rank_key, assignment, candidate, probability)
        if best is None:
            break
        _, assignment, candidate, probability = best
        assignment["courier_ids"].append(candidate["courier_id"])
        assignment["probabilities"].append(probability)
        assignment["total_score"] += candidate["score"]
        used_couriers.add(candidate["courier_id"])
        extra_count += 1
    return assignments


def _duplicate_target(current_success, context):
    if context["low_input_mode"]:
        if current_success < 0.72:
            return DUPLICATE_HIGH_TARGET
        return DUPLICATE_TARGET
    if current_success < 0.70:
        return min(DUPLICATE_HIGH_TARGET, 0.95)
    if current_success < 0.86:
        return min(DUPLICATE_TARGET, 0.92)
    return current_success


def _combined_success_probability(probabilities):
    failure_probability = 1.0
    for probability in probabilities:
        failure_probability *= 1.0 - probability
    return 1.0 - failure_probability


def _clamp_probability(value):
    return min(max(value, 0.0), 1.0)
