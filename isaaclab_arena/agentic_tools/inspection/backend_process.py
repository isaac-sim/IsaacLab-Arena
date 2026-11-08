# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Socket selection and lifecycle helpers for owned inspection backends."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from isaaclab_arena.agentic_tools.inspection.client import InspectorClient
from isaaclab_arena.agentic_tools.inspection.protocol import InspectionError

SIMAPP_SOCKET_ENV = "ARENA_REVIEW_SIMAPP_SOCKET"


def simapp_socket_from_env() -> str | None:
    """Return the configured attach address, if supplied."""
    return os.environ.get(SIMAPP_SOCKET_ENV, "").strip() or None


def select_socket_path(explicit: Path | None = None) -> Path:
    """Select explicit address, environment setting, or checkout-specific default."""
    if explicit is not None:
        return Path(explicit)
    if configured := simapp_socket_from_env():
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[3]
    # Containers mount every clone at the same pathname and share /tmp. Include
    # the bind-mounted directory identity so independent clones cannot collide.
    stat = repo_root.stat()
    checkout_identity = f"{repo_root}:{stat.st_dev}:{stat.st_ino}"
    checkout_id = hashlib.sha256(checkout_identity.encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"arena_scene_inspector_{os.getuid()}_{checkout_id}.sock"


def spawn_backend(socket_path: Path) -> subprocess.Popen[Any]:
    """Launch an owned backend; refuse an existing address without removing it."""
    if os.path.lexists(socket_path):
        raise InspectionError(f"Inspection socket already exists: {socket_path}; clean up stale sockets explicitly.")
    return subprocess.Popen(
        [sys.executable, "-m", "isaaclab_arena.agentic_tools.scene_inspector_backend", "--socket", str(socket_path)],
        env=os.environ.copy(),
    )


def wait_for_backend(
    process: subprocess.Popen[Any],
    socket_path: Path,
    *,
    timeout_s: float = 180.0,
    poll_interval_s: float = 0.5,
) -> None:
    """Wait for an owned child's readiness without changing its socket."""
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        if (exit_code := process.poll()) is not None:
            raise InspectionError(f"Inspection backend exited during boot (exit code: {exit_code}).")
        try:
            InspectorClient(socket_path).ping(timeout_s=min(1.0, max(0.001, deadline - time.monotonic())))
            return
        except InspectionError as exc:
            last_error = exc
        time.sleep(min(poll_interval_s, max(0, deadline - time.monotonic())))
    raise InspectionError(f"Timed out waiting for inspection backend at {socket_path}: {last_error}")


def stop_owned_backend(process: subprocess.Popen[Any], socket_path: Path) -> None:
    """Stop only a live child owned by this launcher, with bounded escalation."""
    if process.poll() is not None:
        return
    # Signal the owned PID, rather than a possibly replaced socket's listener.
    # The server handles SIGTERM with its normal app/socket cleanup.
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
