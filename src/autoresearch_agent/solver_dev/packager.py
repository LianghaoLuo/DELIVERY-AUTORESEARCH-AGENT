"""Solver packaging helpers."""

from __future__ import annotations

import shutil
from pathlib import Path


def package_solver(source_path: str, destination_path: str) -> str:
    """Copy a standalone solver candidate to a solver-side path."""
    source = Path(source_path)
    destination = Path(destination_path)
    if not source.exists():
        raise FileNotFoundError(f"solver source does not exist: {source}")
    if source.resolve() == destination.resolve():
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return str(destination)
