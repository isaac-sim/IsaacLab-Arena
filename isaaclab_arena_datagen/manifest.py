# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone provenance manifest for datagen datasets.

Self-contained: stdlib only, with best-effort optional use of ``torch`` /
``nvidia-smi`` / ``git`` for provenance capture. Imports nothing from the
datagen or evaluation packages, so it can be lifted into a client repo as a
single file. Data-in, JSON-out: build the dataclasses, then call
:func:`write_manifest`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Callable
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclasses.dataclass
class GitInfo:
    """Arena git provenance; fields are ``None`` when git is unavailable."""

    sha: str | None = None
    branch: str | None = None
    dirty: bool | None = None


@dataclasses.dataclass
class GpuInfo:
    """Identity and capacity of a single GPU."""

    index: int
    name: str | None = None
    total_memory_mb: int | None = None
    driver_version: str | None = None


@dataclasses.dataclass
class SystemInfo:
    """Host + GPU provenance captured at run time."""

    hostname: str | None = None
    platform: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    device: str | None = None
    gpus: list[GpuInfo] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SimInfo:
    """Per-job simulation + render settings (these can differ between jobs)."""

    dt: float | None = None
    render_interval: int | None = None
    decimation: int | None = None
    episode_length_s: float | None = None
    render_carb_settings: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SequenceRecord:
    """One collected episode (== one ``dataset.h5``)."""

    episode_index: int
    path: str
    num_frames: int
    camera_ids: list[str]
    dynamic_object_names: list[str]
    outcome: str
    """One of ``"success"`` | ``"failure"`` | ``"timeout"``."""


@dataclasses.dataclass
class JobRecord:
    """One eval job and the sequences it produced."""

    name: str
    status: str
    policy_type: str | None
    policy_config: dict[str, Any]
    language_instruction: str | None
    arena_env_args: list[str]
    datagen_settings: dict[str, Any]
    sim: SimInfo
    num_sequences: int
    num_success: int
    sequences: list[SequenceRecord]


@dataclasses.dataclass
class Manifest:
    """Top-level provenance record for one datagen run."""

    created_at: str
    description: str | None
    generator: dict[str, Any]
    system: SystemInfo
    input_config: Any
    jobs: list[JobRecord]
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Best-effort provenance capture (never raises)
# ---------------------------------------------------------------------------


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        return None


def _run(cmd: list[str]) -> str | None:
    """Run *cmd* and return stripped stdout, or ``None`` on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def capture_git_info(repo_dir: str | None = None) -> GitInfo:
    """Best-effort git SHA/branch/dirty for the repo containing this file (or *repo_dir*)."""
    cwd = repo_dir or os.path.dirname(os.path.abspath(__file__))
    sha = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    branch = _run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "-C", cwd, "status", "--porcelain"])
    dirty = None if status is None else bool(status)
    return GitInfo(sha=sha, branch=branch, dirty=dirty)


def _nvidia_smi_gpus() -> dict[int, dict[str, Any]]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus: dict[int, dict[str, Any]] = {}
    if not out:
        return gpus
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        try:
            mem = int(float(parts[2]))
        except ValueError:
            mem = None
        gpus[idx] = {"name": parts[1], "total_memory_mb": mem, "driver_version": parts[3]}
    return gpus


def capture_system_info(device: str | None = None) -> SystemInfo:
    """Best-effort host + GPU provenance. Missing tools degrade to ``None`` fields."""
    info = SystemInfo(
        hostname=_safe(socket.gethostname),
        platform=_safe(platform.platform),
        python_version=sys.version.split()[0],
        device=device,
    )
    smi = _nvidia_smi_gpus()
    try:
        import torch

        info.torch_version = torch.__version__
        info.cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mb = int(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024))
                info.gpus.append(
                    GpuInfo(
                        index=i,
                        name=torch.cuda.get_device_name(i),
                        total_memory_mb=total_mb,
                        driver_version=smi.get(i, {}).get("driver_version"),
                    )
                )
    except Exception:
        for idx, g in sorted(smi.items()):
            info.gpus.append(
                GpuInfo(
                    index=idx, name=g["name"], total_memory_mb=g["total_memory_mb"], driver_version=g["driver_version"]
                )
            )
    return info


# ---------------------------------------------------------------------------
# Pure record builders
# ---------------------------------------------------------------------------


def clean_datagen_settings(settings: dict) -> dict:
    """Strip layout-only keys (``output_dir``) from a ``DatagenCollectorConfig`` asdict."""
    out = dict(settings)
    out.pop("output_dir", None)
    return out


def relativize_path(path: str, root: str) -> str:
    """Return *path* relative to *root*, or the original path when it is outside *root*."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    if rel.startswith(".."):
        return path
    return rel


def build_job_record(
    *,
    name: str,
    status: str,
    policy_type: str | None,
    policy_config: dict,
    language_instruction: str | None,
    arena_env_args: list[str],
    datagen_settings: dict,
    sim: SimInfo,
    sequence_dicts: list[dict],
    root: str,
) -> JobRecord:
    """Build a :class:`JobRecord` from primitives and the collector's plain-dict sequences."""
    sequences = [
        SequenceRecord(
            episode_index=d["episode_index"],
            path=relativize_path(d["path"], root),
            num_frames=d["num_frames"],
            camera_ids=d["camera_ids"],
            dynamic_object_names=d["dynamic_object_names"],
            outcome=d["outcome"],
        )
        for d in sequence_dicts
    ]
    return JobRecord(
        name=name,
        status=status,
        policy_type=policy_type,
        policy_config=policy_config,
        language_instruction=language_instruction,
        arena_env_args=arena_env_args,
        datagen_settings=datagen_settings,
        sim=sim,
        num_sequences=len(sequences),
        num_success=sum(1 for s in sequences if s.outcome == "success"),
        sequences=sequences,
    )


def build_manifest(
    *,
    created_at: str,
    description: str | None,
    generator_tool: str,
    git: GitInfo,
    system: SystemInfo,
    input_config: Any,
    jobs: list[JobRecord],
) -> Manifest:
    """Assemble the top-level :class:`Manifest`."""
    return Manifest(
        created_at=created_at,
        description=description,
        generator={"tool": generator_tool, "arena_git": dataclasses.asdict(git)},
        system=system,
        input_config=input_config,
        jobs=jobs,
    )


def write_manifest(path: str, manifest: Manifest) -> bool:
    """Write *manifest* to *path* as pretty JSON, atomically. Best-effort: never raises.

    Returns ``True`` on success, ``False`` (after logging a warning) on any failure,
    so a manifest problem can never fail the surrounding run.
    """
    try:
        data = dataclasses.asdict(manifest)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # pragma: no cover - exercised via bad-path test
        print(f"[datagen] Warning: failed to write manifest to {path}: {exc}")
        return False
