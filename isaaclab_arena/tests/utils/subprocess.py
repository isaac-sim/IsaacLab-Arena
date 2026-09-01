# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

_AT_LEAST_ONE_TEST_FAILED = False

_SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_TIMEOUT", "900"))


def run_subprocess(
    cmd,
    env=None,
    timeout_sec: int | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess | None:
    """Run a command in a subprocess with timeout.

    The child is launched with ``start_new_session=True`` so it lives in its
    own process group.  The child-side ``SimulationAppContext`` uses this to
    SIGTERM its entire group before ``os._exit()``, preventing orphaned Kit
    children (shader compiler, GPU workers, …) from holding GPU resources and
    blocking the next subprocess.

    Args:
        cmd: Command to run (list of strings).
        env: Optional environment dict.  Defaults to inheriting the parent env.
        timeout_sec: Per-subprocess wall-clock timeout in seconds.
            Defaults to ``_SUBPROCESS_TIMEOUT_SEC`` (env ``ISAACLAB_ARENA_SUBPROCESS_TIMEOUT``, fallback 900).
        capture_output: If True, capture stdout/stderr and return a
            ``CompletedProcess``.  When False (default) output streams to
            the parent process and the function returns None on success.

    Returns:
        ``CompletedProcess`` when *capture_output* is True, else None.
    """
    if timeout_sec is None:
        timeout_sec = _SUBPROCESS_TIMEOUT_SEC

    print(f"Running command (timeout={timeout_sec}s): {cmd}")
    global _AT_LEAST_ONE_TEST_FAILED

    if env is None:
        env = os.environ.copy()
    env["ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE"] = "1"

    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=timeout_sec,
            capture_output=capture_output,
            text=capture_output,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"\n[isaaclab-arena] Subprocess timed out after {timeout_sec}s\n")
        _AT_LEAST_ONE_TEST_FAILED = True
        raise subprocess.SubprocessError(f"Subprocess timed out after {timeout_sec}s: {cmd}")

    print(f"Command completed with return code: {result.returncode}")
    if result.returncode != 0:
        sys.stderr.write(f"Command failed with return code {result.returncode}\n")
        if capture_output and result.stderr:
            sys.stderr.write(result.stderr)
        _AT_LEAST_ONE_TEST_FAILED = True
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    if capture_output:
        return result
    return None
