"""Candidate-table parsing helpers for agent-side experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    """One feasible courier-task candidate from an reference input case."""

    task_id_list: tuple[str, ...]
    task_id_list_str: str
    courier_id: str
    total_score: float
    willingness: float


@dataclass(frozen=True)
class CandidateTable:
    """Parsed candidate table plus useful lookup indexes."""

    candidates: tuple[Candidate, ...]
    task_ids: frozenset[str]
    courier_ids: frozenset[str]
    candidate_pairs: frozenset[tuple[str, str]]
    candidate_map: dict[tuple[str, str], Candidate] = field(default_factory=dict)

    @property
    def single_task_candidate_count(self) -> int:
        """Return how many rows assign a single order."""
        return sum(1 for candidate in self.candidates if len(candidate.task_id_list) == 1)

    @property
    def bundled_task_candidate_count(self) -> int:
        """Return how many rows assign a two-order bundle."""
        return sum(1 for candidate in self.candidates if len(candidate.task_id_list) > 1)


def normalize_task_id_list(value: object) -> tuple[tuple[str, ...], str]:
    """Normalize a task-list value into tuple form and reference string form."""
    if isinstance(value, str):
        task_ids = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        task_ids = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        task_ids = ()
    return task_ids, ",".join(task_ids)


def parse_candidate_table(input_text: str) -> CandidateTable:
    """Parse reference tab-separated candidate text."""
    candidates: list[Candidate] = []
    task_ids: set[str] = set()
    courier_ids: set[str] = set()
    candidate_pairs: set[tuple[str, str]] = set()
    candidate_map: dict[tuple[str, str], Candidate] = {}

    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            total_score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue

        task_tuple, normalized_task_id_list_str = normalize_task_id_list(
            task_id_list_str
        )
        normalized_courier_id = courier_id.strip()
        if not task_tuple or not normalized_courier_id:
            continue

        candidate = Candidate(
            task_id_list=task_tuple,
            task_id_list_str=normalized_task_id_list_str,
            courier_id=normalized_courier_id,
            total_score=total_score,
            willingness=willingness,
        )
        candidates.append(candidate)
        task_ids.update(task_tuple)
        courier_ids.add(normalized_courier_id)
        pair = (normalized_task_id_list_str, normalized_courier_id)
        candidate_pairs.add(pair)
        candidate_map[pair] = candidate

    return CandidateTable(
        candidates=tuple(candidates),
        task_ids=frozenset(task_ids),
        courier_ids=frozenset(courier_ids),
        candidate_pairs=frozenset(candidate_pairs),
        candidate_map=candidate_map,
    )


def parse_candidate_file(case_path: str | Path) -> CandidateTable:
    """Parse a candidate table from a local file."""
    return parse_candidate_table(Path(case_path).read_text())


def summarize_candidate_table(table: CandidateTable) -> dict[str, int | float]:
    """Return basic case statistics for planning and experiment logs."""
    candidate_count = len(table.candidates)
    single_count = table.single_task_candidate_count
    bundled_count = table.bundled_task_candidate_count
    return {
        "candidate_count": candidate_count,
        "task_count": len(table.task_ids),
        "courier_count": len(table.courier_ids),
        "single_task_candidate_count": single_count,
        "bundled_task_candidate_count": bundled_count,
        "bundled_candidate_ratio": bundled_count / candidate_count
        if candidate_count
        else 0.0,
    }
