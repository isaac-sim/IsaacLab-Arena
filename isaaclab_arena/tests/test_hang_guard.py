# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=60)


def test_exits_with_stack_dump_when_body_is_stuck():
    result = _run_python("""
        import time
        from isaaclab_arena.tests.utils.hang_guard import exit_if_stuck

        with exit_if_stuck(1):
            time.sleep(30)
        """)
    assert result.returncode != 0
    assert "Timeout (0:00:01)" in result.stderr
    assert "most recent call first" in result.stderr


def test_leaves_process_alone_when_body_finishes_in_time():
    result = _run_python("""
        import time
        from isaaclab_arena.tests.utils.hang_guard import exit_if_stuck

        with exit_if_stuck(2):
            pass
        time.sleep(3)
        print("still alive")
        """)
    assert result.returncode == 0
    assert "still alive" in result.stdout
