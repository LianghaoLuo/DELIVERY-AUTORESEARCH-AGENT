"""Solver-output validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autoresearch_agent.solver_dev.parser import (
    CandidateTable,
    normalize_task_id_list,
    parse_candidate_table,
)


@dataclass(frozen=True)
class SolutionAssignment:
    """Normalized solver output row."""

    task_id_list: tuple[str, ...]
    task_id_list_str: str
    courier_ids: tuple[str, ...]


@dataclass
class ValidationResult:
    """Validation result for a solver output."""

    is_valid: bool
    assignments: list[SolutionAssignment] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def assigned_task_count(self) -> int:
        """Return unique order count covered by the output."""
        task_ids: set[str] = set()
        for assignment in self.assignments:
            task_ids.update(assignment.task_id_list)
        return len(task_ids)

    @property
    def used_courier_count(self) -> int:
        """Return unique courier count used by the output."""
        courier_ids: set[str] = set()
        for assignment in self.assignments:
            courier_ids.update(assignment.courier_ids)
        return len(courier_ids)


def is_valid_solution_shape(result: object) -> bool:
    """Check the outer shape expected by the reference solver contract."""
    return validate_solution(result, None).is_valid


def validate_solution(
    result: object,
    candidate_table: CandidateTable | None,
) -> ValidationResult:
    """Validate output shape and hard feasibility constraints."""
    errors: list[str] = []
    warnings: list[str] = []
    assignments: list[SolutionAssignment] = []
    used_couriers: set[str] = set()

    if not isinstance(result, list):
        return ValidationResult(
            is_valid=False,
            errors=["solver result must be a list"],
        )

    for index, item in enumerate(result):
        if not isinstance(item, tuple) or len(item) != 2:
            errors.append(f"row {index}: assignment must be a 2-tuple")
            continue

        task_value, courier_value = item
        task_ids, task_id_list_str = normalize_task_id_list(task_value)
        if not task_ids:
            errors.append(f"row {index}: task_id_list is empty or invalid")
            continue

        if not isinstance(courier_value, (list, tuple)):
            errors.append(f"row {index}: courier list must be a list or tuple")
            continue

        courier_ids = tuple(str(courier).strip() for courier in courier_value)
        courier_ids = tuple(courier for courier in courier_ids if courier)
        if not courier_ids:
            errors.append(f"row {index}: courier list is empty")
            continue

        assignment = SolutionAssignment(
            task_id_list=task_ids,
            task_id_list_str=task_id_list_str,
            courier_ids=courier_ids,
        )
        assignments.append(assignment)

        for courier_id in courier_ids:
            if courier_id in used_couriers:
                errors.append(f"row {index}: courier {courier_id} is assigned twice")
            used_couriers.add(courier_id)

        if candidate_table is not None:
            _validate_against_candidates(
                index=index,
                assignment=assignment,
                candidate_table=candidate_table,
                errors=errors,
                warnings=warnings,
            )

    return ValidationResult(
        is_valid=not errors,
        assignments=assignments,
        errors=errors,
        warnings=warnings,
    )


def validate_solution_for_input(result: object, input_text: str) -> ValidationResult:
    """Parse reference input text and validate a solver result against it."""
    return validate_solution(result, parse_candidate_table(input_text))


def _validate_against_candidates(
    *,
    index: int,
    assignment: SolutionAssignment,
    candidate_table: CandidateTable,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check whether every output courier-task pair exists in the case."""
    for task_id in assignment.task_id_list:
        if task_id not in candidate_table.task_ids:
            errors.append(f"row {index}: unknown task {task_id}")

    for courier_id in assignment.courier_ids:
        if courier_id not in candidate_table.courier_ids:
            errors.append(f"row {index}: unknown courier {courier_id}")

        pair = (assignment.task_id_list_str, courier_id)
        if pair not in candidate_table.candidate_pairs:
            errors.append(
                f"row {index}: candidate pair {assignment.task_id_list_str}/{courier_id} does not exist"
            )

    if len(assignment.courier_ids) > 1:
        warnings.append(
            f"row {index}: duplicate dispatch is present; scoring semantics still need confirmation"
        )


def solution_to_serializable(result: ValidationResult) -> dict[str, Any]:
    """Convert validation details to JSON-serializable primitives."""
    return {
        "is_valid": result.is_valid,
        "assigned_task_count": result.assigned_task_count,
        "used_courier_count": result.used_courier_count,
        "errors": result.errors,
        "warnings": result.warnings,
    }
