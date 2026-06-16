"""Standalone solver entrypoint.

This file is the Python 3.6-compatible boundary for the official judge. It must
not import LangChain, LangGraph, LLM SDKs, or any agent-side modules.
"""


def _parse_candidates(input_text):
    """Parse tab-separated official input text."""
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
        candidates.append(
            (score, task_id_list_str.strip(), courier_id.strip(), willingness)
        )
    return candidates


def solve(input_text):
    """Return a greedy baseline delivery assignment list."""
    candidates = _parse_candidates(input_text)
    candidates.sort(key=lambda item: item[0])

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for score, task_id_list_str, courier_id, willingness in candidates:
        _ = (score, willingness)
        task_ids = [task.strip() for task in task_id_list_str.split(",") if task.strip()]
        if not task_ids or not courier_id:
            continue
        if courier_id in assigned_couriers:
            continue
        if any(task_id in assigned_tasks for task_id in task_ids):
            continue

        assigned_couriers.add(courier_id)
        for task_id in task_ids:
            assigned_tasks.add(task_id)
        result.append((task_id_list_str, [courier_id]))

    return result
