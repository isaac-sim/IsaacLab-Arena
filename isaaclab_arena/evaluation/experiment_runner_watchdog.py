# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stall-restart wrapper around a long-running command (e.g. ``experiment_runner.py``).

A distributed Experiment fails whole when a single Run's process stalls with no output. This
wrapper runs the wrapped command as a subprocess, forwards its output, and treats a silence
longer than ``--stall-timeout-seconds`` as a hang: it kills the process group, clears the
output directory, and relaunches, up to ``--max-restarts`` times. A clean exit (any return
code) is propagated as-is and never restarted.

This is a stop-gap for the underlying stall; see BRANCH_CHANGES.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse watchdog options and the wrapped command (everything after ``--``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stall-timeout-seconds",
        type=float,
        default=300.0,
        help="Restart the command if it emits no output for this many seconds. Default 300 (5 min).",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=3,
        help="Maximum number of stall-triggered restarts before giving up. Default 3.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Directory whose contents are deleted before each restart (the command requires it empty).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=10.0,
        help="How often to check for a stall. Default 10.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to run, preceded by '--' (e.g. -- /isaac-sim/python.sh experiment_runner.py ...).",
    )
    args = parser.parse_args(argv)
    # argparse.REMAINDER keeps the leading '--' separator; drop it so command[0] is the executable.
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    assert args.command, "No command to run. Pass the command after '--'."
    assert args.stall_timeout_seconds > 0, "--stall-timeout-seconds must be positive."
    assert args.max_restarts >= 0, "--max-restarts must be non-negative."
    return args


def _clear_directory_contents(directory: Path) -> None:
    """Delete everything inside ``directory`` while keeping the directory itself."""
    if not directory.exists():
        return
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate the subprocess and its whole process group, escalating to SIGKILL."""
    if process.poll() is not None:
        return
    try:
        group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group_id, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            continue


def _run_once(
    command: list[str], stall_timeout_seconds: float, poll_interval_seconds: float
) -> tuple[int | None, bool]:
    """Run the command once, forwarding output.

    Returns a ``(return_code, stalled)`` pair: on a clean exit ``return_code`` is the process
    exit code and ``stalled`` is False; on a detected stall ``return_code`` is None and
    ``stalled`` is True (the process group has been killed).
    """
    # start_new_session puts the child (and its descendants, e.g. python.sh -> python) in its own
    # process group so a stall can be killed as a group rather than leaking orphaned simulators.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=1,
        text=True,
    )

    last_output_monotonic = time.monotonic()
    output_lock = threading.Lock()

    def _pump_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            nonlocal last_output_monotonic
            with output_lock:
                last_output_monotonic = time.monotonic()
            sys.stdout.write(line)
            sys.stdout.flush()

    pump_thread = threading.Thread(target=_pump_output, daemon=True)
    pump_thread.start()

    try:
        while True:
            time.sleep(poll_interval_seconds)
            return_code = process.poll()
            if return_code is not None:
                pump_thread.join(timeout=10)
                return return_code, False
            with output_lock:
                idle_seconds = time.monotonic() - last_output_monotonic
            if idle_seconds > stall_timeout_seconds:
                print(
                    f"[watchdog] No output for {idle_seconds:.0f}s (limit {stall_timeout_seconds:.0f}s); "
                    "killing the process group.",
                    flush=True,
                )
                _terminate_process_group(process)
                pump_thread.join(timeout=10)
                return None, True
    except KeyboardInterrupt:
        # Watchdog was signalled (e.g. OSMO SIGTERM); take the child group down with us.
        _terminate_process_group(process)
        pump_thread.join(timeout=10)
        raise


def main() -> int:
    args = parse_args()
    command = args.command

    # Forward a watchdog kill signal to the current child by killing our own process group.
    # OSMO cancels the task with SIGTERM; without this the child simulator can outlive us.
    def _forward_signal(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _forward_signal)

    for attempt in range(args.max_restarts + 1):
        if attempt > 0:
            print(
                f"[watchdog] Restart {attempt}/{args.max_restarts} after a stall.",
                flush=True,
            )
            if args.output_directory is not None:
                print(f"[watchdog] Clearing output directory '{args.output_directory}'.", flush=True)
                _clear_directory_contents(args.output_directory)
        print(f"[watchdog] Launching (attempt {attempt + 1}): {' '.join(command)}", flush=True)

        try:
            return_code, stalled = _run_once(command, args.stall_timeout_seconds, args.poll_interval_seconds)
        except KeyboardInterrupt:
            print("[watchdog] Interrupted; terminating any running child and exiting.", flush=True)
            return 130

        if not stalled:
            print(f"[watchdog] Command exited with code {return_code}.", flush=True)
            return return_code if return_code is not None else 0

    print(
        f"[watchdog] Command stalled on every attempt (>{args.max_restarts} restarts); giving up.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
