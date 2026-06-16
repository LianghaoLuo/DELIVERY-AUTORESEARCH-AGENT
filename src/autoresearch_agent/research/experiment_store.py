"""Persistent experiment logging for AutoResearch runs."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoresearch_agent.solver_dev.runner import (
    SolverRunResult,
    SolverSuiteResult,
    solver_run_to_serializable,
    solver_suite_to_serializable,
)
from autoresearch_agent.solver_dev.variants import (
    VariantBatchResult,
    VariantSuiteBatchResult,
    variant_batch_to_serializable,
    variant_suite_batch_to_serializable,
)


@dataclass(kw_only=True)
class ExperimentStore:
    """Append-only JSONL and Markdown experiment store."""

    root_dir: Path = Path("experiments")
    jsonl_filename: str = "runs.jsonl"
    markdown_filename: str = "runs.md"

    @property
    def jsonl_path(self) -> Path:
        """Return the JSONL log path."""
        return self.root_dir / self.jsonl_filename

    @property
    def markdown_path(self) -> Path:
        """Return the Markdown summary path."""
        return self.root_dir / self.markdown_filename

    def append_solver_run(
        self,
        result: SolverRunResult,
        *,
        label: str,
        notes: str = "",
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one solver run and return the record."""
        record = {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "label": label,
            "notes": notes,
            "run": solver_run_to_serializable(result),
        }
        _add_provenance(record, provenance)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_markdown_summary(record)
        return record

    def append_variant_batch(
        self,
        batch: VariantBatchResult,
        *,
        label: str,
        baseline_result: SolverRunResult,
        notes: str = "",
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one solver-variant batch run and return the record."""
        record = {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "label": label,
            "notes": notes,
            "case_path": batch.case_path,
            "best_variant_path": batch.best_variant_path,
            "baseline_metrics": solver_run_to_serializable(baseline_result)["metrics"],
            "batch": variant_batch_to_serializable(batch),
        }
        _add_provenance(record, provenance)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_variant_batch_markdown(record)
        return record

    def append_solver_suite_run(
        self,
        result: SolverSuiteResult,
        *,
        label: str,
        notes: str = "",
        provenance: dict[str, Any] | None = None,
        baseline_suite: dict[str, Any] | None = None,
        evidence_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one solver suite run and return the record."""
        record = {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "label": label,
            "notes": notes,
            "baseline_suite": baseline_suite or {},
            "suite": solver_suite_to_serializable(result),
        }
        _add_provenance(record, provenance)
        _add_evidence_profile(record, evidence_profile)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_suite_markdown_summary(record)
        return record

    def append_variant_suite_batch(
        self,
        batch: VariantSuiteBatchResult,
        *,
        label: str,
        baseline_result: SolverSuiteResult,
        notes: str = "",
        provenance: dict[str, Any] | None = None,
        baseline_suite: dict[str, Any] | None = None,
        evidence_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one solver-variant suite batch run and return the record."""
        record = {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "label": label,
            "notes": notes,
            "best_variant_path": batch.best_variant_path,
            "baseline_suite": baseline_suite
            or solver_suite_to_serializable(baseline_result),
            "baseline_aggregate_metrics": solver_suite_to_serializable(baseline_result)[
                "aggregate_metrics"
            ],
            "batch": variant_suite_batch_to_serializable(batch),
        }
        _add_provenance(record, provenance)
        _add_evidence_profile(record, evidence_profile)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_variant_suite_batch_markdown(record)
        return record

    def load_records(self) -> list[dict[str, Any]]:
        """Load all JSONL experiment records."""
        if not self.jsonl_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.jsonl_path.open(encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    loaded = json.loads(stripped)
                    if isinstance(loaded, dict):
                        records.append(loaded)
        return records

    def _append_markdown_summary(self, record: dict[str, Any]) -> None:
        """Append a compact human-readable summary."""
        if not self.markdown_path.exists():
            self.markdown_path.write_text("# Experiment Runs\n\n", encoding="utf-8")

        run = record["run"]
        metrics = run["metrics"]
        validation = run["validation"]
        lines = [
            f"## {record['label']} - {record['created_at']}",
            "",
            f"- Run ID: `{record['run_id']}`",
            f"- Solver: `{run['solver_path']}`",
            f"- Case: `{run['case_path']}`",
            f"- Valid: `{validation['is_valid']}`",
            f"- Timed out: `{run['timed_out']}`",
            f"- Elapsed seconds: `{run['elapsed_seconds']:.6f}`",
            f"- Assigned tasks: `{metrics['assigned_task_count']}`",
            f"- Unassigned tasks: `{metrics['unassigned_task_count']}`",
            f"- Total score: `{metrics['total_score']:.6f}`",
            f"- Average willingness: `{metrics['average_willingness']:.6f}`",
            f"- Proxy score: `{metrics['proxy_score']:.6f}`",
        ]
        if record["notes"]:
            lines.append(f"- Notes: {record['notes']}")
        lines.extend(_provenance_lines(record))
        if validation["errors"]:
            lines.append(f"- Errors: `{validation['errors']}`")
        lines.append("")

        with self.markdown_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))

    def _append_suite_markdown_summary(self, record: dict[str, Any]) -> None:
        """Append a compact human-readable suite summary."""
        if not self.markdown_path.exists():
            self.markdown_path.write_text("# Experiment Runs\n\n", encoding="utf-8")

        suite = record["suite"]
        aggregate = suite["aggregate_metrics"]
        lines = [
            f"## {record['label']} - {record['created_at']}",
            "",
            f"- Run ID: `{record['run_id']}`",
            f"- Solver: `{suite['solver_path']}`",
            f"- Suite valid: `{aggregate['is_valid']}`",
            f"- Cases: `{aggregate['case_count']}`",
            f"- Invalid cases: `{aggregate['invalid_case_count']}`",
            f"- Timeout cases: `{aggregate['timeout_count']}`",
            f"- Mean coverage: `{aggregate['mean_task_coverage_ratio']:.6f}`",
            f"- Mean expected success: `{aggregate['mean_expected_success_ratio']:.6f}`",
            f"- Mean proxy score: `{aggregate['mean_proxy_score']:.6f}`",
            f"- Worst case: `{aggregate['worst_case_id']}`",
        ]
        if record["notes"]:
            lines.append(f"- Notes: {record['notes']}")
        lines.extend(_provenance_lines(record))
        lines.append("")

        with self.markdown_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))

    def _append_variant_batch_markdown(self, record: dict[str, Any]) -> None:
        """Append a human-readable variant leaderboard."""
        if not self.markdown_path.exists():
            self.markdown_path.write_text("# Experiment Runs\n\n", encoding="utf-8")

        batch = record["batch"]
        lines = [
            f"## {record['label']} - {record['created_at']}",
            "",
            f"- Run ID: `{record['run_id']}`",
            f"- Case: `{record['case_path']}`",
            f"- Best variant: `{record['best_variant_path']}`",
            "",
            "| Rank | Variant | Valid | Assigned | Total Score | Avg Willingness | Proxy Score |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for result in batch["variant_results"]:
            metrics = result["metrics"]
            lines.append(
                "| "
                f"{result['rank']} | "
                f"`{result['variant_path']}` | "
                f"{metrics['is_valid']} | "
                f"{metrics['assigned_task_count']} | "
                f"{metrics['total_score']:.6f} | "
                f"{metrics['average_willingness']:.6f} | "
                f"{metrics['proxy_score']:.6f} |"
            )
        if record["notes"]:
            lines.extend(["", f"- Notes: {record['notes']}"])
        lines.extend(_provenance_lines(record))
        lines.append("")

        with self.markdown_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))

    def _append_variant_suite_batch_markdown(self, record: dict[str, Any]) -> None:
        """Append a human-readable variant suite leaderboard."""
        if not self.markdown_path.exists():
            self.markdown_path.write_text("# Experiment Runs\n\n", encoding="utf-8")

        batch = record["batch"]
        lines = [
            f"## {record['label']} - {record['created_at']}",
            "",
            f"- Run ID: `{record['run_id']}`",
            f"- Best variant: `{record['best_variant_path']}`",
            "",
            "| Rank | Variant | Valid | Cases | Timeouts | Mean Coverage | Mean Expected Success | Mean Proxy | Worst Case |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for result in batch["variant_results"]:
            aggregate = result["aggregate_metrics"]
            lines.append(
                "| "
                f"{result['rank']} | "
                f"`{result['variant_path']}` | "
                f"{aggregate['is_valid']} | "
                f"{aggregate['case_count']} | "
                f"{aggregate['timeout_count']} | "
                f"{aggregate['mean_task_coverage_ratio']:.6f} | "
                f"{aggregate['mean_expected_success_ratio']:.6f} | "
                f"{aggregate['mean_proxy_score']:.6f} | "
                f"`{aggregate['worst_case_id']}` |"
            )
        if record["notes"]:
            lines.extend(["", f"- Notes: {record['notes']}"])
        lines.extend(_provenance_lines(record))
        lines.append("")

        with self.markdown_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))


def _add_provenance(
    record: dict[str, Any],
    provenance: dict[str, Any] | None,
) -> None:
    """Attach optional provenance metadata to one experiment record."""
    if provenance:
        record["provenance"] = dict(provenance)


def _add_evidence_profile(
    record: dict[str, Any],
    evidence_profile: dict[str, Any] | None,
) -> None:
    """Attach optional failure-mode evidence metadata to one experiment record."""
    if evidence_profile:
        record["evidence_profile"] = dict(evidence_profile)


def _provenance_lines(record: dict[str, Any]) -> list[str]:
    """Render compact provenance fields for human-readable logs."""
    provenance = record.get("provenance", {})
    if not isinstance(provenance, dict) or not provenance:
        return []
    lines = [
        f"- Origin: `{provenance.get('origin', '')}`",
        f"- Agent version: `{provenance.get('agent_version', '')}`",
    ]
    action = provenance.get("action", "")
    if action:
        lines.append(f"- Agent action: `{action}`")
    selected_by = provenance.get("selected_by", "")
    if selected_by:
        lines.append(f"- Selected by: `{selected_by}`")
    accepted_source = provenance.get("accepted_source", "")
    if accepted_source and accepted_source != selected_by:
        lines.append(f"- Accepted source: `{accepted_source}`")
    return lines
