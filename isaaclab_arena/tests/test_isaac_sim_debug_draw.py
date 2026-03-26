# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for IsaacSimDebugDraw."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


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


def test_isaac_sim_debug_draw_smoke():
    """Smoke test: IsaacSimDebugDraw initializes and runs without errors."""
    result = run_simulation_app_function(smoke_test_debug_draw)
    assert result, "IsaacSimDebugDraw smoke test failed"
