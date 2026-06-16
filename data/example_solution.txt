"""
示例 Solver（贪心基线）
========================
选手代码必须定义 solve(input_text: str) -> list 函数。
返回格式：[(task_id_list_str, [courier_id, ...]), ...]

本文件可直接用于本地测试：
    python3 judge_server.py --test example_solver.py --case small_seed100.txt
"""

import heapq
from collections import defaultdict


def solve(input_text: str) -> list:
    """
    贪心基线：按 total_score 升序，依次分配最优骑手。
    
    输入：制表符分隔的文本（含表头）
    输出：[(task_id_list_str, [courier_id, ...]), ...]
    """
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    # 解析候选集
    candidates = []  # (score, task_id_list_str, courier_id, willingness)
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
        candidates.append((score, task_id_list_str.strip(), courier_id.strip(), willingness))

    # 按分数升序排列
    candidates.sort(key=lambda x: x[0])

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for score, task_id_list_str, courier_id, willingness in candidates:
        task_ids = [t.strip() for t in task_id_list_str.split(",")]

        # 跳过已分配的骑手或订单
        if courier_id in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue

        # 分配
        assigned_couriers.add(courier_id)
        for t in task_ids:
            assigned_tasks.add(t)
        result.append((task_id_list_str, [courier_id]))

    return result
