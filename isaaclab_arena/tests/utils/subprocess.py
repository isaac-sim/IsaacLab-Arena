# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

_AT_LEAST_ONE_TEST_FAILED = False

_SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_TIMEOUT", "900"))
_SUBPROCESS_STARTUP_TIMEOUT_SEC = int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_STARTUP_TIMEOUT", "120"))
_SUBPROCESS_STARTUP_RETRIES = int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_STARTUP_RETRIES", "2"))
_SUBPROCESS_TIMEOUT_RETRIES = int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_TIMEOUT_RETRIES", "1"))

_APP_LAUNCHER_COMPLETE_MARKER = b"AppLauncher initialization complete"


@dataclass
class _AttemptResult:
    result: subprocess.CompletedProcess[str]
    kill_reason: str | None
    elapsed_sec: float


def _forward_output(chunk: bytes, parent_stream) -> None:
    """Forward a binary child-output chunk to a parent text stream."""
    parent_buffer = getattr(parent_stream, "buffer", None)
    if parent_buffer is not None:
        parent_buffer.write(chunk)
        parent_buffer.flush()
    else:
        parent_stream.write(chunk.decode(errors="replace"))
        parent_stream.flush()


def _kill_process_group(process: subprocess.Popen, process_group_id: int) -> None:
    """Kill the subprocess and every descendant that remains in its process group."""
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()


def _run_subprocess_attempt(
    cmd,
    env,
    timeout_sec: int,
    capture_output: bool,
    startup_timeout_sec: int,
) -> _AttemptResult:
    """Run one isolated attempt while watching AppLauncher startup."""
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    process_group_id = process.pid
    stdout_data = bytearray()
    stderr_data = bytearray()
    streams = {
        process.stdout.fileno(): (stdout_data, sys.stdout),
        process.stderr.fileno(): (stderr_data, sys.stderr),
    }
    start_time = time.monotonic()
    startup_complete = startup_timeout_sec <= 0
    kill_reason = None

    try:
        while process.poll() is None:
            if streams:
                ready_fds, _, _ = select.select(list(streams), [], [], 0.1)
                for file_descriptor in ready_fds:
                    try:
                        chunk = os.read(file_descriptor, 65536)
                    except OSError:
                        streams.pop(file_descriptor, None)
                        continue
                    if not chunk:
                        streams.pop(file_descriptor, None)
                        continue
                    output, parent_stream = streams[file_descriptor]
                    output.extend(chunk)
                    if not capture_output:
                        _forward_output(chunk, parent_stream)
            else:
                time.sleep(0.1)

            if process.poll() is not None:
                break

            elapsed_sec = time.monotonic() - start_time
            if not startup_complete and (
                _APP_LAUNCHER_COMPLETE_MARKER in stdout_data or _APP_LAUNCHER_COMPLETE_MARKER in stderr_data
            ):
                startup_complete = True

            if not startup_complete and elapsed_sec > startup_timeout_sec:
                kill_reason = "startup hang"
                break
            if elapsed_sec > timeout_sec:
                kill_reason = "timeout"
                break
    finally:
        # The session leader can exit while Kit workers are still alive and holding
        # stdout/stderr or GPU resources. Always clean the entire group before returning.
        _kill_process_group(process, process_group_id)

    try:
        remaining_stdout, remaining_stderr = process.communicate(timeout=5)
        stdout_data.extend(remaining_stdout)
        stderr_data.extend(remaining_stderr)
        if not capture_output:
            _forward_output(remaining_stdout, sys.stdout)
            _forward_output(remaining_stderr, sys.stderr)
    except subprocess.TimeoutExpired:
        _kill_process_group(process, process_group_id)
        with contextlib.suppress(subprocess.TimeoutExpired):
            remaining_stdout, remaining_stderr = process.communicate(timeout=1)
            stdout_data.extend(remaining_stdout)
            stderr_data.extend(remaining_stderr)
            if not capture_output:
                _forward_output(remaining_stdout, sys.stdout)
                _forward_output(remaining_stderr, sys.stderr)

    elapsed_sec = time.monotonic() - start_time
    returncode = process.returncode if process.returncode is not None else -signal.SIGKILL
    result = subprocess.CompletedProcess(
        cmd,
        returncode,
        stdout_data.decode(errors="replace"),
        stderr_data.decode(errors="replace"),
    )
    return _AttemptResult(result=result, kill_reason=kill_reason, elapsed_sec=elapsed_sec)


def _write_captured_output(result: subprocess.CompletedProcess[str]) -> None:
    """Write captured child output when an attempt ultimately fails."""
    if result.stdout:
        sys.stderr.write("\n[isaaclab-arena] Captured subprocess stdout:\n")
        sys.stderr.write(result.stdout)
    if result.stderr:
        sys.stderr.write("\n[isaaclab-arena] Captured subprocess stderr:\n")
        sys.stderr.write(result.stderr)


def run_subprocess(
    cmd,
    env=None,
    timeout_sec: int | None = None,
    capture_output: bool = False,
    check: bool = True,
    startup_timeout_sec: int = _SUBPROCESS_STARTUP_TIMEOUT_SEC,
    startup_retries: int = _SUBPROCESS_STARTUP_RETRIES,
    timeout_retries: int = _SUBPROCESS_TIMEOUT_RETRIES,
) -> subprocess.CompletedProcess[str] | None:
    """Run an Isaac Sim command with bounded startup and hard-timeout retries.

    Each attempt runs in a new process group. The entire group is killed after
    the attempt, including orphaned Kit workers that could otherwise block the
    next test. A process that does not print the AppLauncher completion marker
    within the startup deadline is treated as a transient startup hang.

    Args:
        cmd: Command to run (list of strings).
        env: Optional environment dict. Defaults to inheriting the parent env.
        timeout_sec: Per-attempt wall-clock timeout in seconds.
        capture_output: Capture and return stdout/stderr instead of streaming it.
        check: Raise ``CalledProcessError`` for a nonzero exit code.
        startup_timeout_sec: Seconds to wait for AppLauncher initialization. Use
            zero to disable the startup watchdog.
        startup_retries: Number of retries allowed after a startup hang.
        timeout_retries: Number of retries allowed after the hard timeout.

    Returns:
        ``CompletedProcess`` when *capture_output* is True, else None.
    """
    if timeout_sec is None:
        timeout_sec = _SUBPROCESS_TIMEOUT_SEC
    assert startup_retries >= 0
    assert timeout_retries >= 0

    global _AT_LEAST_ONE_TEST_FAILED

    child_env = (os.environ if env is None else env).copy()
    child_env["ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE"] = "1"
    retries_remaining = {
        "startup hang": startup_retries,
        "timeout": timeout_retries,
    }
    attempt_number = 0

    while True:
        attempt_number += 1
        print(f"Running command (attempt={attempt_number}, timeout={timeout_sec}s): {cmd}", flush=True)
        attempt = _run_subprocess_attempt(
            cmd,
            child_env,
            timeout_sec,
            capture_output,
            startup_timeout_sec,
        )

        if attempt.kill_reason is not None:
            reason = attempt.kill_reason
            sys.stderr.write(
                f"\n[isaaclab-arena] Subprocess {reason} after {attempt.elapsed_sec:.1f}s (attempt {attempt_number})\n"
            )
            if retries_remaining[reason] > 0:
                retries_remaining[reason] -= 1
                sys.stderr.write(
                    "[isaaclab-arena] Retrying in a clean process group "
                    f"({retries_remaining[reason]} {reason} retries remain)\n"
                )
                continue

            if capture_output:
                _write_captured_output(attempt.result)
            _AT_LEAST_ONE_TEST_FAILED = True
            raise subprocess.SubprocessError(f"Subprocess {reason} after {attempt.elapsed_sec:.1f}s: {cmd}")

        result = attempt.result
        print(f"Command completed with return code: {result.returncode}", flush=True)
        if check and result.returncode != 0:
            sys.stderr.write(f"Command failed with return code {result.returncode}\n")
            if capture_output:
                _write_captured_output(result)
            _AT_LEAST_ONE_TEST_FAILED = True
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        if capture_output:
            return result
        return None
