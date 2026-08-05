# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for IsaacSimDebugDraw."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.isaac_sim_debug_draw import (
    AXIS_X_COLOR,
    AXIS_Y_COLOR,
    AXIS_Z_COLOR,
    axis_length_from_extents,
    local_xyz_axis_segments,
)


def test_local_xyz_axis_segments_rgb_from_origin():
    """Axis helpers emit red/green/blue segments of the requested length."""
    starts, ends, colors = local_xyz_axis_segments(
        (1.0, 2.0, 3.0),
        x_dir=(2.0, 0.0, 0.0),
        y_dir=(0.0, 3.0, 0.0),
        z_dir=(0.0, 0.0, 4.0),
        length_m=0.5,
    )
    assert starts == [(1.0, 2.0, 3.0)] * 3
    assert ends[0] == (1.5, 2.0, 3.0)
    assert ends[1] == (1.0, 2.5, 3.0)
    assert ends[2] == (1.0, 2.0, 3.5)
    assert colors == [AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR]


def test_local_xyz_axis_segments_skips_degenerate_directions():
    """Degenerate axis directions are omitted instead of remapped to +X."""
    starts, ends, colors = local_xyz_axis_segments(
        (0.0, 0.0, 0.0),
        x_dir=(1.0, 0.0, 0.0),
        y_dir=(0.0, 0.0, 0.0),
        z_dir=(0.0, 0.0, 1.0),
        length_m=1.0,
    )
    assert starts == [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    assert ends == [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    assert colors == [AXIS_X_COLOR, AXIS_Z_COLOR]


def test_axis_length_from_extents_uses_fraction_with_minimum():
    """Axis length is a fraction of the largest extent, never below the minimum."""
    assert axis_length_from_extents((1.0, 2.0, 0.5), fraction=0.35, min_m=0.05) == 0.7
    assert axis_length_from_extents((0.01, 0.02, 0.01), fraction=0.35, min_m=0.05) == 0.05


def test_overlay_rgb_segments_on_png_draws_opaque_lines():
    """PNG overlay draws always-on-top segments without needing Kit."""
    import io

    from PIL import Image

    from isaaclab_arena.utils.isaac_sim_debug_draw import overlay_rgb_segments_on_png

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(buf, format="PNG")
    overlaid = overlay_rgb_segments_on_png(
        buf.getvalue(),
        [
            ((8, 32), (56, 32), (255, 0, 0)),
            ((32, 8), (32, 56), (0, 255, 0)),
        ],
        width_px=2,
    )
    with Image.open(io.BytesIO(overlaid)) as image:
        assert image.getpixel((32, 32)) in {(255, 0, 0), (0, 255, 0)}
        assert image.getpixel((40, 32)) == (255, 0, 0)
        assert image.getpixel((32, 40)) == (0, 255, 0)


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
    result = run_function_with_persistent_simulation_app(smoke_test_debug_draw)
    assert result, "IsaacSimDebugDraw smoke test failed"
