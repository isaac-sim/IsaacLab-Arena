# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Software metadata collection for relation benchmark runs."""

from __future__ import annotations

import platform
import subprocess
import torch
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
"""Repository root used for git metadata collection."""

GIT_TIMEOUT_SECONDS = 5
"""Maximum time allowed for one local Git command."""


@dataclass(frozen=True)
class SoftwareMetadata:
    """Software identity for the checkout and runtime orchestrating a run."""

    git_commit: str | None
    """Checked-out git commit, if available."""

    git_dirty: bool | None
    """Whether the checkout has changes, or None when git is unavailable."""

    python_version: str
    """Python runtime version."""

    pytorch_version: str
    """PyTorch runtime version."""

    cuda_version: str | None
    """CUDA version reported by PyTorch."""


def _git_output(repository_root: Path, *args: str) -> str:
    """Return stdout from one git command."""
    return subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    ).stdout


def collect_software_metadata(repository_root: Path = REPOSITORY_ROOT) -> SoftwareMetadata:
    """Collect software identity without requiring an available git checkout."""
    git_commit = None
    git_dirty = None
    try:
        repository_root = repository_root.resolve()
        git_root = Path(_git_output(repository_root, "rev-parse", "--show-toplevel").strip()).resolve()
        if git_root == repository_root:
            git_commit = _git_output(repository_root, "rev-parse", "HEAD").strip()
            git_dirty = bool(_git_output(repository_root, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return SoftwareMetadata(
        git_commit=git_commit or None,
        git_dirty=git_dirty,
        python_version=platform.python_version(),
        pytorch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
    )
