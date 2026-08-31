# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for IsaacSimDebugDraw."""

import os
import subprocess

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def smoke_test_debug_draw(simulation_app) -> bool:
    """Verify IsaacSimDebugDraw can be instantiated and basic methods work."""
    from isaaclab_arena.utils.isaac_sim_debug_draw import IsaacSimDebugDraw

    # Test instantiation
    debug_draw = IsaacSimDebugDraw()

    # Test drawing a bounding box
    debug_draw.draw_bbox(
        min_point=(0.0, 0.0, 0.0),
        max_point=(1.0, 1.0, 1.0),
    )

    # Test clearing
    debug_draw.clear()

    return True


@pytest.mark.with_subprocess
def test_isaac_sim_debug_draw_smoke():
    """Smoke test: IsaacSimDebugDraw initializes and runs without errors."""
    result = subprocess.run(
        [TestConstants.python_path, __file__],
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_TIMEOUT", "900")),
        start_new_session=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    result = run_function_with_persistent_simulation_app(smoke_test_debug_draw)
    raise SystemExit(0 if result else 1)
