"""Local solver runner for agent-side experiments."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import multiprocessing
import queue
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

from autoresearch_agent.solver_dev.case_suite import SolverCase
from autoresearch_agent.solver_dev.metrics import (
    SolutionMetrics,
    calculate_solution_metrics,
    metrics_to_serializable,
)
from autoresearch_agent.solver_dev.parser import parse_candidate_table
from autoresearch_agent.solver_dev.validator import (
    ValidationResult,
    solution_to_serializable,
    validate_solution,
)


@dataclass(frozen=True)
class SolverRunResult:
    """Result of running a solver against one local case."""

    solver_path: str
    case_path: str
    case_id: str
    case_weight: float
    elapsed_seconds: float
    timed_out: bool
    process_exit_code: int | None
    stdout: str
    stderr: str
    solver_sha256: str
    output: object
    validation: ValidationResult
    metrics: SolutionMetrics
    error: str = ""


@dataclass(frozen=True)
class SuiteAggregateMetrics:
    """Aggregate local metrics for one solver across a case suite."""

    is_valid: bool
    case_count: int
    valid_case_count: int
    invalid_case_count: int
    timeout_count: int
    total_elapsed_seconds: float
    mean_proxy_score: float
    worst_proxy_score: float
    mean_risk_adjusted_proxy_score: float
    mean_expected_success_ratio: float
    mean_task_coverage_ratio: float
    worst_case_id: str


@dataclass(frozen=True)
class SolverSuiteResult:
    """Result of running a solver against multiple local cases."""

    solver_path: str
    results: list[SolverRunResult]
    aggregate_metrics: SuiteAggregateMetrics


def run_solver_case(
    solver_path: str,
    case_path: str,
    *,
    timeout_seconds: float = 10.0,
) -> SolverRunResult:
    """Run a solver against one case and validate the output."""
    case_path_obj = Path(case_path)
    return run_solver_text(
        solver_path,
        case_path_obj.read_text(),
        case_id=case_path_obj.stem,
        timeout_seconds=timeout_seconds,
        case_path=str(case_path_obj),
    )


def run_solver_text(
    solver_path: str,
    input_text: str,
    case_id: str,
    *,
    timeout_seconds: float = 10.0,
    case_path: str = "",
    case_weight: float = 1.0,
) -> SolverRunResult:
    """Run a solver against in-memory case text and validate the output."""
    solver_path_obj = Path(solver_path)
    candidate_table = parse_candidate_table(input_text)
    solver_sha256 = _file_sha256(solver_path_obj)

    start = time.perf_counter()
    execution = _execute_solver_in_subprocess(
        solver_path_obj,
        input_text,
        timeout_seconds=timeout_seconds,
    )
    elapsed_seconds = time.perf_counter() - start
    output = execution["output"]
    error = str(execution["error"])
    timed_out = bool(execution["timed_out"]) or elapsed_seconds > timeout_seconds

    validation = validate_solution(output, candidate_table)
    if error:
        validation.errors.append(error)
        validation.is_valid = False
    if timed_out and not any("timed out" in item.lower() for item in validation.errors):
        validation.errors.append(
            f"solver timed out after {timeout_seconds:.3f} seconds"
        )
        validation.is_valid = False
    metrics = calculate_solution_metrics(
        validation,
        candidate_table,
        case_id=case_id,
        timed_out=timed_out,
    )

    return SolverRunResult(
        solver_path=str(solver_path_obj),
        case_path=case_path or case_id,
        case_id=case_id,
        case_weight=case_weight,
        elapsed_seconds=elapsed_seconds,
        timed_out=timed_out,
        process_exit_code=cast(int | None, execution["process_exit_code"]),
        stdout=str(execution["stdout"]),
        stderr=str(execution["stderr"]),
        solver_sha256=solver_sha256,
        output=output,
        validation=validation,
        metrics=metrics,
        error=error,
    )


def run_solver_suite(
    solver_path: str,
    cases: list[SolverCase],
    *,
    timeout_seconds: float = 10.0,
) -> SolverSuiteResult:
    """Run a solver against a deterministic local case suite."""
    results = [
        run_solver_text(
            solver_path,
            case.input_text,
            case_id=case.case_id,
            timeout_seconds=timeout_seconds,
            case_path=case.source_name or case.case_id,
            case_weight=case.weight,
        )
        for case in cases
    ]
    return SolverSuiteResult(
        solver_path=str(Path(solver_path)),
        results=results,
        aggregate_metrics=_aggregate_suite_metrics(results),
    )


def _load_solver_module(solver_path: Path) -> ModuleType:
    """Load a solver file as an isolated Python module."""
    module_name = f"_solver_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solver module from {solver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_solve(module: ModuleType) -> Callable[[str], Any]:
    """Return the solve callable from a loaded solver module."""
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise AttributeError("solver module must define callable solve(input_text)")
    return cast(Callable[[str], Any], solve)


def _execute_solver_in_subprocess(
    solver_path: Path,
    input_text: str,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    """Execute solver code in a child process with a hard timeout."""
    context = _multiprocessing_context()
    result_queue: multiprocessing.Queue[dict[str, object]] = context.Queue(maxsize=1)
    process = context.Process(
        target=_solver_worker,
        args=(str(solver_path), input_text, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():  # pragma: no cover - defensive hard stop
            process.kill()
            process.join(1.0)
        return {
            "output": [],
            "error": f"TimeoutError: solver exceeded {timeout_seconds:.3f} seconds",
            "stdout": "",
            "stderr": "",
            "timed_out": True,
            "process_exit_code": process.exitcode,
        }

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        payload = {
            "output": [],
            "error": "ProcessError: solver process exited without returning a result",
            "stdout": "",
            "stderr": "",
        }
    payload["timed_out"] = False
    payload["process_exit_code"] = process.exitcode
    return payload


def _solver_worker(
    solver_path: str,
    input_text: str,
    result_queue: multiprocessing.Queue[dict[str, object]],
) -> None:
    """Run solver.solve inside a child process and return captured output."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    output: object = []
    error = ""
    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        try:
            module = _load_solver_module(Path(solver_path))
            solve = _get_solve(module)
            output = solve(input_text)
        except Exception as exc:  # pragma: no cover - preserved for experiment logs
            error = f"{type(exc).__name__}: {exc}"
    result_queue.put(
        {
            "output": output,
            "error": error,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
        }
    )


def _multiprocessing_context() -> Any:
    """Return a multiprocessing context that works well for local experiments."""
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-Unix fallback
        return multiprocessing.get_context("spawn")


def _aggregate_suite_metrics(results: list[SolverRunResult]) -> SuiteAggregateMetrics:
    """Aggregate per-case solver metrics for ranking and reporting."""
    case_count = len(results)
    valid_case_count = sum(1 for result in results if result.metrics.is_valid)
    timeout_count = sum(1 for result in results if result.timed_out)
    invalid_case_count = case_count - valid_case_count
    total_elapsed_seconds = sum(result.elapsed_seconds for result in results)
    worst = max(
        results,
        key=lambda result: result.metrics.proxy_score,
        default=None,
    )
    return SuiteAggregateMetrics(
        is_valid=case_count > 0 and invalid_case_count == 0 and timeout_count == 0,
        case_count=case_count,
        valid_case_count=valid_case_count,
        invalid_case_count=invalid_case_count,
        timeout_count=timeout_count,
        total_elapsed_seconds=total_elapsed_seconds,
        mean_proxy_score=_weighted_mean(
            [(result.metrics.proxy_score, result.case_weight) for result in results]
        ),
        worst_proxy_score=worst.metrics.proxy_score if worst else 0.0,
        mean_risk_adjusted_proxy_score=_weighted_mean(
            [
                (result.metrics.risk_adjusted_proxy_score, result.case_weight)
                for result in results
            ]
        ),
        mean_expected_success_ratio=_weighted_mean(
            [
                (result.metrics.expected_success_ratio, result.case_weight)
                for result in results
            ]
        ),
        mean_task_coverage_ratio=_weighted_mean(
            [
                (result.metrics.task_coverage_ratio, result.case_weight)
                for result in results
            ]
        ),
        worst_case_id=worst.case_id if worst else "",
    )


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    """Return the weighted mean for a possibly empty list."""
    total_weight = sum(weight for _, weight in values)
    if not values or total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight


def _file_sha256(path: Path) -> str:
    """Return a stable content hash for the solver file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solver_run_to_serializable(result: SolverRunResult) -> dict[str, Any]:
    """Convert a solver run result to JSON-serializable primitives."""
    output_count = len(result.output) if isinstance(result.output, list) else 0
    output_preview = result.output[:5] if isinstance(result.output, list) else []
    return {
        "solver_path": result.solver_path,
        "case_path": result.case_path,
        "case_id": result.case_id,
        "case_weight": result.case_weight,
        "elapsed_seconds": result.elapsed_seconds,
        "timed_out": result.timed_out,
        "process_exit_code": result.process_exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "solver_sha256": result.solver_sha256,
        "error": result.error,
        "output_count": output_count,
        "output_preview": output_preview,
        "validation": solution_to_serializable(result.validation),
        "metrics": metrics_to_serializable(result.metrics),
    }


def suite_aggregate_to_serializable(
    aggregate: SuiteAggregateMetrics,
) -> dict[str, Any]:
    """Convert suite aggregate metrics to JSON-serializable primitives."""
    return {
        "is_valid": aggregate.is_valid,
        "case_count": aggregate.case_count,
        "valid_case_count": aggregate.valid_case_count,
        "invalid_case_count": aggregate.invalid_case_count,
        "timeout_count": aggregate.timeout_count,
        "total_elapsed_seconds": aggregate.total_elapsed_seconds,
        "mean_proxy_score": aggregate.mean_proxy_score,
        "worst_proxy_score": aggregate.worst_proxy_score,
        "mean_risk_adjusted_proxy_score": aggregate.mean_risk_adjusted_proxy_score,
        "mean_expected_success_ratio": aggregate.mean_expected_success_ratio,
        "mean_task_coverage_ratio": aggregate.mean_task_coverage_ratio,
        "worst_case_id": aggregate.worst_case_id,
    }


def solver_suite_to_serializable(result: SolverSuiteResult) -> dict[str, Any]:
    """Convert a suite run result to JSON-serializable primitives."""
    return {
        "solver_path": result.solver_path,
        "aggregate_metrics": suite_aggregate_to_serializable(result.aggregate_metrics),
        "case_results": [
            solver_run_to_serializable(case_result) for case_result in result.results
        ],
    }
