# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys

from isaaclab_arena.tests.utils.subprocess import run_subprocess


def test_run_subprocess_captures_output_without_startup_watchdog():
    result = run_subprocess(
        [sys.executable, "-c", "print('subprocess output')"],
        capture_output=True,
        startup_timeout_sec=0,
    )

    assert result is not None
    assert result.returncode == 0
    assert result.stdout == "subprocess output\n"


def test_run_subprocess_retries_startup_hang(tmp_path):
    attempt_file = tmp_path / "startup-attempt"
    script = """
import pathlib
import sys
import time

attempt_file = pathlib.Path(sys.argv[1])
attempt = int(attempt_file.read_text()) + 1 if attempt_file.exists() else 1
attempt_file.write_text(str(attempt))
if attempt == 1:
    time.sleep(10)
print('[isaaclab-arena] AppLauncher initialization complete', file=sys.stderr, flush=True)
"""

    result = run_subprocess(
        [sys.executable, "-c", script, str(attempt_file)],
        capture_output=True,
        timeout_sec=5,
        startup_timeout_sec=1,
        startup_retries=1,
        timeout_retries=0,
    )

    assert result is not None
    assert result.returncode == 0
    assert attempt_file.read_text() == "2"


def test_run_subprocess_retries_hard_timeout(tmp_path):
    attempt_file = tmp_path / "timeout-attempt"
    script = """
import pathlib
import sys
import time

attempt_file = pathlib.Path(sys.argv[1])
attempt = int(attempt_file.read_text()) + 1 if attempt_file.exists() else 1
attempt_file.write_text(str(attempt))
print('[isaaclab-arena] AppLauncher initialization complete', file=sys.stderr, flush=True)
if attempt == 1:
    time.sleep(10)
"""

    result = run_subprocess(
        [sys.executable, "-c", script, str(attempt_file)],
        capture_output=True,
        timeout_sec=1,
        startup_timeout_sec=1,
        startup_retries=0,
        timeout_retries=1,
    )

    assert result is not None
    assert result.returncode == 0
    assert attempt_file.read_text() == "2"
