"""Generated pressure-targeted combo solver.

Family: pressure_targeted
Strategy type: pressure_combo
Params: {'alpha': 92.5, 'budget_high_risk_target': 0.98, 'budget_max_couriers_per_assignment': 4, 'budget_max_extra_dispatches': 60, 'budget_mid_risk_target': 0.95, 'budget_min_roi': -45.0, 'expected_failed_weight': 0.0, 'expected_success_credit': 100.0, 'high_risk_target': 0.95, 'high_success_min_roi': 15.0, 'low_willingness_alpha': 90.0, 'low_willingness_assignment_ratio': 0.2, 'low_willingness_success_threshold': 0.9, 'max_couriers_per_assignment': 3, 'max_extra_dispatches': 40, 'max_merge_passes': 1, 'merge_min_improvement': 0.0, 'mid_risk_target': 0.92, 'min_roi': -25.0, 'profile': 'scarce_b050_lowwill_c4', 'scarce_alpha': 95.0, 'scarce_bundle_bonus': 50.0, 'scarce_max_extra_dispatches': 0, 'scarce_ratio_threshold': 1.35, 'scarce_use_merge': False, 'score_weight': 1.0, 'time_budget_seconds': 8.5}
Python 3.6-compatible candidate solver. Do not import agent-side modules here.
"""

import time


ALPHA = 92.5
LOW_WILLINGNESS_ALPHA = 90.0
SCORE_WEIGHT = 1.0
EXPECTED_FAILED_WEIGHT = 0.0
EXPECTED_SUCCESS_CREDIT = 100.0
BUNDLE_BIAS = 0.0
UNASSIGNED_TASK_WEIGHT = 10000.0
MAX_PASSES = 1
TIME_BUDGET_SECONDS = 8.5
SCARCE_ALPHA = 95.0
SCARCE_BUNDLE_BONUS = 50.0
SCARCE_USE_MERGE = False
MAX_MERGE_PASSES = 1
MERGE_MIN_IMPROVEMENT = 0.0
HIGH_RISK_TARGET = 0.95
MID_RISK_TARGET = 0.92
MIN_ROI = -25.0
HIGH_SUCCESS_MIN_ROI = 15.0
MAX_EXTRA_DISPATCHES = 40
MAX_COURIERS_PER_ASSIGNMENT = 3
SCARCE_RATIO_THRESHOLD = 1.35
LOW_WILLINGNESS_SUCCESS_THRESHOLD = 0.9
LOW_WILLINGNESS_ASSIGNMENT_RATIO = 0.2
INPUT_LOW_WILLINGNESS_BEST_THRESHOLD = 0.90
INPUT_LOW_WILLINGNESS_AVERAGE_THRESHOLD = 0.90
INPUT_LOW_WILLINGNESS_TASK_RATIO = 0.25
SCARCE_MAX_EXTRA_DISPATCHES = 0
SCARCE_MIN_ROI_BONUS = 12.0
SCARCE_COURIER_PENALTY = 18.0
NORMAL_COURIER_PENALTY = 4.0
LOWWILL_BUDGET_LOW_SUCCESS_THRESHOLD = 0.72
LOWWILL_BUDGET_HIGH_RISK_TARGET = 0.98
LOWWILL_BUDGET_MID_RISK_TARGET = 0.95
LOWWILL_BUDGET_MAX_EXTRA_DISPATCHES = 60
LOWWILL_BUDGET_MAX_COURIERS = 4
LOWWILL_BUDGET_MIN_ROI = -45.0


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
        task_ids = [task.strip() for task in task_id_list_str.split(",") if task.strip()]
        if not task_ids or not courier_id.strip():
            continue
        candidates.append({
            "score": score,
            "task_id_list_str": ",".join(task_ids),
            "task_ids": task_ids,
            "courier_id": courier_id.strip(),
            "willingness": willingness,
        })
    return candidates


def solve(input_text):
    """Return a guarded assignment with a scarce-courier bundle-first branch."""
    started_at = time.time()
    candidates = _parse_candidates(input_text)
    task_ids = set(task_id for candidate in candidates for task_id in candidate["task_ids"])
    courier_ids = set(candidate["courier_id"] for candidate in candidates)
    task_count = len(task_ids)
    courier_count = len(courier_ids)
    scarce_mode = _is_scarce_mode(task_count, courier_count)
    if scarce_mode:
        candidates.sort(key=_scarce_candidate_sort_key)
        selected = _assign_greedily(candidates)
        selected = _improve(selected, candidates, task_count, started_at)
        if SCARCE_USE_MERGE:
            selected = _merge_singles(selected, candidates, task_count, started_at)
        return [
            (candidate["task_id_list_str"], [candidate["courier_id"]])
            for candidate in selected
        ]

    alpha = _select_alpha(candidates)
    candidates.sort(key=lambda candidate: (
        candidate["score"]
        - alpha * candidate["willingness"]
        + (BUNDLE_BIAS if len(candidate["task_ids"]) > 1 else 0.0),
        candidate["score"],
    ))
    selected = _assign_greedily(candidates)
    selected = _improve(selected, candidates, task_count, started_at)
    assignments = _to_assignments(selected)
    assignments = _augment_duplicates(assignments, candidates, task_count, courier_count, started_at)
    return [
        (assignment["task_id_list_str"], list(assignment["courier_ids"]))
        for assignment in assignments
    ]


def _scarce_candidate_sort_key(candidate):
    task_size = float(len(candidate["task_ids"]) or 1)
    bundle_bonus = SCARCE_BUNDLE_BONUS if len(candidate["task_ids"]) > 1 else 0.0
    return (
        candidate["score"] / task_size
        - SCARCE_ALPHA * candidate["willingness"]
        - bundle_bonus,
        0 if len(candidate["task_ids"]) > 1 else 1,
        candidate["score"],
    )



def _assign_greedily(candidates):
    assigned_couriers = set()
    assigned_tasks = set()
    selected = []
    for candidate in candidates:
        if candidate["courier_id"] in assigned_couriers:
            continue
        if any(task_id in assigned_tasks for task_id in candidate["task_ids"]):
            continue
        selected.append(candidate)
        assigned_couriers.add(candidate["courier_id"])
        for task_id in candidate["task_ids"]:
            assigned_tasks.add(task_id)
    return selected


def _select_alpha(candidates):
    if LOW_WILLINGNESS_ALPHA != ALPHA and _is_input_low_willingness_mode(candidates):
        return LOW_WILLINGNESS_ALPHA
    return ALPHA


def _is_input_low_willingness_mode(candidates):
    best_by_task = {}
    for candidate in candidates:
        willingness = _clamp_probability(candidate["willingness"])
        for task_id in candidate["task_ids"]:
            if willingness > best_by_task.get(task_id, 0.0):
                best_by_task[task_id] = willingness
    if not best_by_task:
        return False
    values = list(best_by_task.values())
    low_count = sum(1 for value in values if value < INPUT_LOW_WILLINGNESS_BEST_THRESHOLD)
    low_ratio = float(low_count) / float(len(values))
    average_best = sum(values) / float(len(values))
    return (
        low_ratio >= INPUT_LOW_WILLINGNESS_TASK_RATIO
        or average_best < INPUT_LOW_WILLINGNESS_AVERAGE_THRESHOLD
    )


def _improve(selected, candidates, task_count, started_at):
    current_score = _solution_objective(selected, task_count)
    for _ in range(MAX_PASSES):
        changed = False
        for index in range(len(selected)):
            if time.time() - started_at > TIME_BUDGET_SECONDS * 0.60:
                return selected
            occupied_tasks, occupied_couriers = _occupied_without(selected, index)
            best_candidate = selected[index]
            best_score = current_score
            for candidate in candidates:
                if candidate["courier_id"] in occupied_couriers:
                    continue
                if any(task_id in occupied_tasks for task_id in candidate["task_ids"]):
                    continue
                trial = list(selected)
                trial[index] = candidate
                trial_score = _solution_objective(trial, task_count)
                if trial_score < best_score:
                    best_score = trial_score
                    best_candidate = candidate
            if best_candidate is not selected[index]:
                selected[index] = best_candidate
                current_score = best_score
                changed = True
        if not changed:
            break
    return selected


def _occupied_without(selected, excluded_index):
    tasks = set()
    couriers = set()
    for index, candidate in enumerate(selected):
        if index == excluded_index:
            continue
        couriers.add(candidate["courier_id"])
        for task_id in candidate["task_ids"]:
            tasks.add(task_id)
    return tasks, couriers


def _to_assignments(selected):
    assignments = []
    for candidate in selected:
        willingness = _clamp_probability(candidate["willingness"])
        assignments.append({
            "task_id_list_str": candidate["task_id_list_str"],
            "task_ids": candidate["task_ids"],
            "courier_ids": [candidate["courier_id"]],
            "probabilities": [willingness],
            "total_score": candidate["score"],
        })
    return assignments


def _augment_duplicates(assignments, candidates, task_count, courier_count, started_at):
    used_couriers = set()
    for assignment in assignments:
        for courier_id in assignment["courier_ids"]:
            used_couriers.add(courier_id)

    candidates_by_task_list = {}
    for candidate in candidates:
        candidates_by_task_list.setdefault(candidate["task_id_list_str"], []).append(candidate)

    scarce_mode = _is_scarce_mode(task_count, courier_count)
    low_willingness_mode = _is_low_willingness_mode(assignments)
    max_extra = _effective_budget_max_extra_dispatches(scarce_mode, low_willingness_mode)
    extra_dispatches = 0
    while extra_dispatches < max_extra:
        if time.time() - started_at > TIME_BUDGET_SECONDS:
            break
        best = None
        used_ratio = float(len(used_couriers)) / float(courier_count or 1)
        for assignment in assignments:
            current_success = _combined_success_probability(assignment["probabilities"])
            if low_willingness_mode and not scarce_mode:
                target, max_couriers, min_roi = _budget_risk_policy(current_success)
            else:
                target, max_couriers, min_roi = _risk_policy(
                    current_success,
                    scarce_mode,
                    low_willingness_mode,
                )
            if len(assignment["courier_ids"]) >= max_couriers:
                continue
            if current_success >= target:
                continue
            for candidate in candidates_by_task_list.get(assignment["task_id_list_str"], []):
                if candidate["courier_id"] in used_couriers:
                    continue
                probability = _clamp_probability(candidate["willingness"])
                gain_per_task = (1.0 - current_success) * probability
                success_gain = gain_per_task * len(assignment["task_ids"])
                raw_roi = (
                    (EXPECTED_FAILED_WEIGHT + EXPECTED_SUCCESS_CREDIT) * success_gain
                    - SCORE_WEIGHT * candidate["score"]
                )
                if raw_roi < min_roi:
                    continue
                scarcity_penalty = _courier_scarcity_penalty(scarce_mode, used_ratio)
                ranked_roi = raw_roi - scarcity_penalty
                risk_gap = max(target - current_success, 0.0) * len(assignment["task_ids"])
                tail_bonus = 8.0 if current_success < LOWWILL_BUDGET_LOW_SUCCESS_THRESHOLD else 0.0
                rank_key = (
                    risk_gap + tail_bonus,
                    ranked_roi,
                    raw_roi,
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
        extra_dispatches += 1
    return assignments


def _effective_budget_max_extra_dispatches(scarce_mode, low_willingness_mode):
    if scarce_mode:
        return min(MAX_EXTRA_DISPATCHES, SCARCE_MAX_EXTRA_DISPATCHES)
    if low_willingness_mode:
        return LOWWILL_BUDGET_MAX_EXTRA_DISPATCHES
    return min(MAX_EXTRA_DISPATCHES, 20)


def _budget_risk_policy(current_success):
    if current_success < 0.70:
        return (
            LOWWILL_BUDGET_HIGH_RISK_TARGET,
            LOWWILL_BUDGET_MAX_COURIERS,
            LOWWILL_BUDGET_MIN_ROI,
        )
    if current_success < 0.86:
        return (
            LOWWILL_BUDGET_MID_RISK_TARGET,
            min(LOWWILL_BUDGET_MAX_COURIERS, 3),
            LOWWILL_BUDGET_MIN_ROI + 10.0,
        )
    return (max(current_success, MID_RISK_TARGET), 2, HIGH_SUCCESS_MIN_ROI)



def _risk_policy(current_success, scarce_mode, low_willingness_mode):
    if current_success < 0.70:
        max_couriers = 2 if scarce_mode else MAX_COURIERS_PER_ASSIGNMENT
        return HIGH_RISK_TARGET, max_couriers, MIN_ROI + (SCARCE_MIN_ROI_BONUS if scarce_mode else 0.0)
    if current_success < 0.85:
        return MID_RISK_TARGET, 2, MIN_ROI + (SCARCE_MIN_ROI_BONUS if scarce_mode else 0.0)
    if low_willingness_mode and not scarce_mode:
        return max(current_success, MID_RISK_TARGET), 2, HIGH_SUCCESS_MIN_ROI
    return current_success, 1, HIGH_SUCCESS_MIN_ROI


def _is_scarce_mode(task_count, courier_count):
    if task_count <= 0:
        return False
    return float(courier_count) / float(task_count) <= SCARCE_RATIO_THRESHOLD


def _is_low_willingness_mode(assignments):
    if not assignments:
        return False
    low_count = 0
    total_success = 0.0
    for assignment in assignments:
        success = _combined_success_probability(assignment["probabilities"])
        total_success += success
        if success < LOW_WILLINGNESS_SUCCESS_THRESHOLD:
            low_count += 1
    low_ratio = float(low_count) / float(len(assignments))
    average_success = total_success / float(len(assignments))
    return low_ratio >= LOW_WILLINGNESS_ASSIGNMENT_RATIO or average_success < 0.88


def _effective_max_extra_dispatches(scarce_mode, low_willingness_mode):
    if scarce_mode:
        return min(MAX_EXTRA_DISPATCHES, SCARCE_MAX_EXTRA_DISPATCHES)
    if low_willingness_mode:
        return MAX_EXTRA_DISPATCHES
    return min(MAX_EXTRA_DISPATCHES, 20)


def _courier_scarcity_penalty(scarce_mode, used_ratio):
    if scarce_mode:
        return SCARCE_COURIER_PENALTY * used_ratio
    return NORMAL_COURIER_PENALTY * used_ratio


def _solution_objective(selected, task_count):
    covered_tasks = set()
    total_score = 0.0
    expected_success = 0.0
    for candidate in selected:
        total_score += candidate["score"]
        if len(candidate["task_ids"]) > 1:
            total_score += BUNDLE_BIAS
        willingness = _clamp_probability(candidate["willingness"])
        for task_id in candidate["task_ids"]:
            if task_id not in covered_tasks:
                expected_success += willingness
                covered_tasks.add(task_id)
    unassigned = max(task_count - len(covered_tasks), 0)
    expected_failed = max(task_count - expected_success, 0.0)
    return (
        SCORE_WEIGHT * total_score
        + EXPECTED_FAILED_WEIGHT * expected_failed
        - EXPECTED_SUCCESS_CREDIT * expected_success
        + UNASSIGNED_TASK_WEIGHT * unassigned
    )


def _combined_success_probability(probabilities):
    failure_probability = 1.0
    for probability in probabilities:
        failure_probability *= 1.0 - probability
    return 1.0 - failure_probability


def _clamp_probability(value):
    return min(max(value, 0.0), 1.0)
