# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for IsaacSimDebugDraw."""

import math
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox


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


def test_draw_object_bboxes_uses_oriented_corners():
    """Object drawing emits the exact OBB corners in wireframe edge order."""
    from isaaclab_arena.utils.isaac_sim_debug_draw import DEFAULT_COLOR, IsaacSimDebugDraw

    half_yaw = math.pi / 8
    bbox = OrientedBoundingBox(
        (1.0, 2.0, 3.0),
        (2.0, 1.0, 0.5),
        (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
    )
    obj = SimpleNamespace(get_world_bounding_box=lambda: bbox)
    debug_draw = IsaacSimDebugDraw.__new__(IsaacSimDebugDraw)
    debug_draw._draw = MagicMock()

    debug_draw.draw_object_bboxes([obj], thickness=2.0)

    corners = bbox.get_corners()[0].tolist()
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    debug_draw._draw.draw_lines.assert_called_once_with(
        [corners[start] for start, _ in edges],
        [corners[end] for _, end in edges],
        [DEFAULT_COLOR] * 12,
        [2.0] * 12,
    )


if __name__ == "__main__":
    result = run_function_with_persistent_simulation_app(smoke_test_debug_draw)
    raise SystemExit(0 if result else 1)
