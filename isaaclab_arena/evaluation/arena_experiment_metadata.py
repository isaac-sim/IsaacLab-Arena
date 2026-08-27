# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persist reproducibility metadata for one Experiment Runner process."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg


ARENA_EXPERIMENT_METADATA_FILENAME = "arena_experiment_metadata.json"
"""Filename containing the resolved configuration and process metadata for an Experiment."""

ARENA_EXPERIMENT_METADATA_SCHEMA_VERSION = 1
"""Schema version for ``arena_experiment_metadata.json``."""


def _utc_timestamp() -> str:
    """Return the current UTC time in an ISO 8601 representation."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _absolute_path(path: str | Path) -> str:
    """Return a stable absolute representation without requiring the path to exist."""
    return str(Path(path).expanduser().resolve(strict=False))


def _sha256_file(path: Path) -> str | None:
    """Return the file's SHA-256 digest, or ``None`` when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _run_git(repo_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a short read-only Git query without allowing it to affect the Experiment."""
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_output(result: subprocess.CompletedProcess[str] | None) -> str | None:
    """Return stripped stdout from a successful Git query."""
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def _git_dirty(repo_root: Path) -> bool | None:
    """Return whether a Git worktree has tracked or untracked changes."""
    status_result = _run_git(repo_root, ["status", "--porcelain", "--untracked-files=normal"])
    return bool(status_result.stdout.strip()) if status_result is not None and status_result.returncode == 0 else None


def _resolve_source_revision() -> dict[str, Any]:
    """Resolve Arena and Isaac Lab revisions and worktree states when Git metadata is available."""
    environment_revision = next(
        (
            os.environ.get(variable_name)
            for variable_name in ("ISAACLAB_ARENA_GIT_COMMIT", "GIT_COMMIT", "CI_COMMIT_SHA", "GITHUB_SHA")
            if os.environ.get(variable_name)
        ),
        None,
    )
    repo_root = Path(__file__).resolve().parents[2]
    git_revision = _git_output(_run_git(repo_root, ["rev-parse", "HEAD"]))
    isaaclab_submodule_root = repo_root / "submodules" / "IsaacLab"
    isaaclab_revision = _git_output(_run_git(isaaclab_submodule_root, ["rev-parse", "HEAD"]))
    isaaclab_gitlink_result = _git_output(_run_git(repo_root, ["ls-tree", "HEAD", "submodules/IsaacLab"]))
    isaaclab_gitlink_revision = (
        isaaclab_gitlink_result.split()[2]
        if isaaclab_gitlink_result is not None and len(isaaclab_gitlink_result.split()) >= 3
        else None
    )
    return {
        "git_commit": environment_revision or git_revision,
        "git_dirty": _git_dirty(repo_root),
        "isaaclab_submodule": {
            "git_commit": isaaclab_revision,
            "expected_git_commit": isaaclab_gitlink_revision,
            "matches_superproject": (
                isaaclab_revision == isaaclab_gitlink_revision
                if isaaclab_revision is not None and isaaclab_gitlink_revision is not None
                else None
            ),
            "git_dirty": _git_dirty(isaaclab_submodule_root),
        },
    }


def _namespace_value(arguments: argparse.Namespace, *names: str) -> Any:
    """Return the first available argparse value among several compatible field names."""
    for name in names:
        if hasattr(arguments, name):
            return getattr(arguments, name)
    return None


def _json_safe(value: Any) -> Any:
    """Convert common configuration values to deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _qualified_type_name(value: Any) -> str:
    """Return the import-qualified type name for a resolved configuration object."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _resolved_run_metadata(run_cfg: ArenaRunCfg) -> dict[str, Any]:
    """Build the benchmark-relevant subset of one resolved Run configuration."""
    environment_builder = run_cfg.environment_builder
    rollout_limit = run_cfg.rollout_limit
    cameras_enabled = bool(getattr(run_cfg.environment, "enable_cameras", False))
    camera_height = getattr(environment_builder, "camera_height", None)
    camera_width = getattr(environment_builder, "camera_width", None)
    resolution_is_explicit = camera_height is not None and camera_width is not None
    camera_resolution_source = None
    if cameras_enabled:
        camera_resolution_source = "environment_builder_override" if resolution_is_explicit else "embodiment_default"

    return {
        "environment_config_type": _qualified_type_name(run_cfg.environment),
        "policy_config_type": _qualified_type_name(run_cfg.policy),
        "num_envs": environment_builder.num_envs,
        "device": environment_builder.device,
        "seed": environment_builder.seed,
        "placement_seed": environment_builder.placement_seed,
        "camera": {
            "enabled": cameras_enabled,
            "height": camera_height if resolution_is_explicit else None,
            "width": camera_width if resolution_is_explicit else None,
            "resolution_source": camera_resolution_source,
        },
        "rollout_limit": {
            "num_steps": rollout_limit.num_steps,
            "num_episodes": rollout_limit.num_episodes,
            "source": (
                "configured"
                if rollout_limit.num_steps is not None or rollout_limit.num_episodes is not None
                else "policy"
            ),
        },
        "num_rebuilds": run_cfg.num_rebuilds,
        "variations": _json_safe(run_cfg.variations),
    }


def _resolved_runs_metadata(experiment_cfg: ArenaExperimentCfg) -> dict[str, dict[str, Any]]:
    """Build metadata for every resolved Run while isolating per-Run collection failures."""
    resolved_runs = {}
    for run_name, run_cfg in experiment_cfg.runs.items():
        try:
            resolved_runs[run_name] = _resolved_run_metadata(run_cfg)
        except Exception as error:
            resolved_runs[run_name] = {
                "metadata_error": {
                    "type": type(error).__name__,
                    "message": str(error)[:2000],
                }
            }
    return resolved_runs


class ArenaExperimentMetadataRecorder:
    """Write lifecycle and resolved configuration metadata without failing the Experiment."""

    def __init__(self, output_path: Path, metadata: dict[str, Any]) -> None:
        self.output_path = output_path
        self._metadata = metadata

    @classmethod
    def start(
        cls,
        experiment_output_directory: str | Path,
        experiment_config_path: str | Path,
        experiment_overrides: list[str],
        args_cli: argparse.Namespace,
        command: list[str] | None = None,
    ) -> ArenaExperimentMetadataRecorder:
        """Create and immediately persist the initial metadata for an Experiment invocation."""
        output_directory = Path(experiment_output_directory)
        config_path = Path(experiment_config_path)
        metadata: dict[str, Any] = {
            "schema_version": ARENA_EXPERIMENT_METADATA_SCHEMA_VERSION,
            "status": "starting",
            "started_at": None,
            "resolved_at": None,
            "finished_at": None,
            "runs": {},
        }
        recorder = cls(output_directory / ARENA_EXPERIMENT_METADATA_FILENAME, metadata)
        try:
            metadata.update({
                "started_at": _utc_timestamp(),
                "experiment_config": {
                    "path": _absolute_path(config_path),
                    "format": config_path.suffix.lower().lstrip("."),
                    "sha256": _sha256_file(config_path),
                },
                "experiment_output_directory": _absolute_path(output_directory),
                "working_directory": _absolute_path(Path.cwd()),
                "command": list(command) if command is not None else [sys.executable, *sys.argv],
                "experiment_overrides": list(experiment_overrides),
                "runtime": {
                    "device": _namespace_value(args_cli, "device"),
                    "headless": _namespace_value(args_cli, "headless"),
                    "visualizer": _namespace_value(args_cli, "visualizer", "viz"),
                    "rendering_mode": _namespace_value(args_cli, "rendering_mode"),
                    "renderer": _namespace_value(args_cli, "renderer"),
                    "enable_cameras": _namespace_value(args_cli, "enable_cameras"),
                    "record_viewport_video": _namespace_value(args_cli, "record_viewport_video"),
                    "record_camera_video": _namespace_value(args_cli, "record_camera_video"),
                    "environment_render_mode": (
                        "rgb_array" if bool(_namespace_value(args_cli, "record_viewport_video")) else None
                    ),
                },
            })
        except Exception as error:
            metadata["metadata_collection_error"] = {
                "type": type(error).__name__,
                "message": str(error)[:2000],
            }
            recorder._warn(error)
        try:
            metadata["revision"] = _resolve_source_revision()
        except Exception as error:
            metadata["revision"] = {
                "metadata_error": {
                    "type": type(error).__name__,
                    "message": str(error)[:2000],
                }
            }
            recorder._warn(error)
        recorder._write_best_effort()
        return recorder

    def record_resolved_experiment(self, experiment_cfg: ArenaExperimentCfg) -> None:
        """Persist resolved per-Run configuration before execution begins."""
        try:
            self._metadata["status"] = "running"
            self._metadata["resolved_at"] = _utc_timestamp()
            self._metadata["runs"] = _resolved_runs_metadata(experiment_cfg)
            self._write_best_effort()
        except Exception as error:
            self._warn(error)

    def finish(self, status: str, error: BaseException | None = None) -> None:
        """Persist the final process status and optional error summary."""
        try:
            self._metadata["status"] = status
            self._metadata["finished_at"] = _utc_timestamp()
            if error is not None:
                self._metadata["error"] = {
                    "type": type(error).__name__,
                    "message": str(error)[:2000],
                }
            else:
                self._metadata.pop("error", None)
            self._write_best_effort()
        except Exception as metadata_error:
            self._warn(metadata_error)

    def _write_best_effort(self) -> None:
        """Atomically replace the artifact, warning instead of raising when persistence fails."""
        temporary_path = self.output_path.with_name(f".{self.output_path.name}.tmp")
        try:
            contents = json.dumps(_json_safe(self._metadata), allow_nan=False, indent=2) + "\n"
            temporary_path.write_text(contents, encoding="utf-8")
            temporary_path.replace(self.output_path)
        except Exception as error:
            with suppress(OSError):
                temporary_path.unlink()
            self._warn(error)

    def _warn(self, error: BaseException) -> None:
        """Report a metadata failure without allowing warning output to affect execution."""
        with suppress(Exception):
            print(f"[WARN] Could not write Arena Experiment metadata to {self.output_path}: {error}", file=sys.stderr)
