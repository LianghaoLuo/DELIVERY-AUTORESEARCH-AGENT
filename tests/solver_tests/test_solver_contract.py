import importlib.util
from pathlib import Path

from autoresearch_agent.solver_dev.parser import parse_candidate_table
from autoresearch_agent.solver_dev.validator import validate_solution


def load_solver_module(solver_path: str = "solvers/solver.py"):
    solver_path = Path(solver_path)
    spec = importlib.util.spec_from_file_location("solver_boundary", solver_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_solver_defines_solve() -> None:
    module = load_solver_module()

    assert callable(module.solve)


def test_solver_returns_list_for_reference_example_data() -> None:
    module = load_solver_module()
    input_text = Path("data/large_seed301.txt").read_text()

    result = module.solve(input_text)

    assert isinstance(result, list)


def test_solver_baseline_output_is_valid_for_reference_example_data() -> None:
    module = load_solver_module()
    input_text = Path("data/large_seed301.txt").read_text()
    candidate_table = parse_candidate_table(input_text)

    result = module.solve(input_text)
    validation = validate_solution(result, candidate_table)

    assert validation.is_valid, validation.errors
    assert validation.assigned_task_count > 0
    assert validation.used_courier_count == len(result)


def test_prior_solver_output_is_valid_for_reference_example_data() -> None:
    module = load_solver_module("solvers/prior_solver.py")
    input_text = Path("data/large_seed301.txt").read_text()
    candidate_table = parse_candidate_table(input_text)

    result = module.solve(input_text)
    validation = validate_solution(result, candidate_table)

    assert validation.is_valid, validation.errors
    assert validation.assigned_task_count == len(candidate_table.task_ids)
