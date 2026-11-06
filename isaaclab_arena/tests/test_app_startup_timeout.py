# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
    STARTUP_BEGIN_MARKER,
    STARTUP_COMPLETE_MARKER,
    STARTUP_TIMEOUT_ENV_VAR,
)

# Replaces AppLauncher with a native call that never returns and keeps the GIL held,
# as a Kit startup hang can.
_HUNG_STARTUP_SCRIPT = """
import argparse
import ctypes

from isaaclab_arena.utils.isaaclab_utils import simulation_app

simulation_app.AppLauncher = lambda _: ctypes.PyDLL(None).sleep(600)
simulation_app.get_app_launcher(argparse.Namespace())
"""


def test_hung_app_startup_exits_with_stack_dump():
    env = dict(os.environ)
    env[STARTUP_TIMEOUT_ENV_VAR] = "2"
    result = subprocess.run(
        [TestConstants.python_path, "-c", _HUNG_STARTUP_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 1, result.stderr
    assert STARTUP_BEGIN_MARKER in result.stderr
    assert STARTUP_COMPLETE_MARKER not in result.stderr
    assert "Timeout (0:00:02)!" in result.stderr
    assert "in get_app_launcher" in result.stderr
