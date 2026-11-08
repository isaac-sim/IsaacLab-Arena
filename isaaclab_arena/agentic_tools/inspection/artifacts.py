# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read completed inspection artifacts and release owned preview outputs."""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator
from pathlib import Path

from isaaclab_arena.agentic_tools.inspection.kit_viewport import sim_preview_cache_dir
from isaaclab_arena.agentic_tools.inspection.protocol import InspectionError, InspectionResult, validate_result


def _temporary_directory(result: InspectionResult) -> Path | None:
    raw = result["temporary_dir"]
    if raw is None:
        return None
    path = Path(raw)
    root = sim_preview_cache_dir().resolve()
    # Only direct preview job directories may be recursively deleted; never the root or a symlink.
    if path.is_symlink() or path.resolve().parent != root or not path.name.startswith("preview-"):
        raise InspectionError(f"Refusing to remove unowned preview directory: {path}")
    return path.resolve()


@contextlib.contextmanager
def consume_result(result: InspectionResult) -> Iterator[InspectionResult]:
    """Release this result's temporary preview directory, including on failure."""
    validate_result(result)
    directory = _temporary_directory(result)
    try:
        yield result
    finally:
        if directory is not None:
            shutil.rmtree(directory)


def _artifact_path(result: InspectionResult, artifact_id: str) -> Path:
    try:
        path = Path(result["artifacts"][artifact_id]["path"])
    except KeyError as exc:
        raise InspectionError(f"Missing inspection artifact: {artifact_id}") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise InspectionError(f"Missing or empty inspection artifact {artifact_id}: {path}")
    return path


def read_artifact(result: InspectionResult, artifact_id: str) -> bytes:
    """Read a completed nonempty artifact; missing files are errors."""
    try:
        return _artifact_path(result, artifact_id).read_bytes()
    except OSError as exc:
        raise InspectionError(f"Cannot read inspection artifact {artifact_id}: {exc}") from exc


def copy_artifact(result: InspectionResult, artifact_id: str, destination: Path) -> Path:
    """Copy a completed artifact outside its owned temporary directory."""
    source = _artifact_path(result, artifact_id)
    directory = _temporary_directory(result)
    destination = destination.resolve()
    if directory is not None and destination.is_relative_to(directory):
        raise InspectionError("Artifact destination must be outside the temporary preview directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
