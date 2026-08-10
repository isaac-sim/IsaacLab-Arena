# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for review-GUI axis overlay helpers."""

from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.axis_overlay import (
    AXIS_X_COLOR,
    AXIS_Y_COLOR,
    AXIS_Z_COLOR,
    axis_length_from_extents,
    local_xyz_axis_segments,
    overlay_rgb_segments_on_png,
)


def test_local_xyz_axis_segments_rgb_from_origin():
    """Axis helpers emit red/green/blue segments of the requested length."""
    segments = local_xyz_axis_segments(
        (1.0, 2.0, 3.0),
        x_dir=(2.0, 0.0, 0.0),
        y_dir=(0.0, 3.0, 0.0),
        z_dir=(0.0, 0.0, 4.0),
        length_m=0.5,
    )
    assert [start for start, _end, _color in segments] == [(1.0, 2.0, 3.0)] * 3
    assert segments[0][1] == (1.5, 2.0, 3.0)
    assert segments[1][1] == (1.0, 2.5, 3.0)
    assert segments[2][1] == (1.0, 2.0, 3.5)
    assert [color for _start, _end, color in segments] == [AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR]


def test_local_xyz_axis_segments_skips_degenerate_directions():
    """Degenerate axis directions are omitted instead of remapped to +X."""
    segments = local_xyz_axis_segments(
        (0.0, 0.0, 0.0),
        x_dir=(1.0, 0.0, 0.0),
        y_dir=(0.0, 0.0, 0.0),
        z_dir=(0.0, 0.0, 1.0),
        length_m=1.0,
    )
    assert [start for start, _end, _color in segments] == [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    assert [end for _start, end, _color in segments] == [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    assert [color for _start, _end, color in segments] == [AXIS_X_COLOR, AXIS_Z_COLOR]


def test_axis_length_from_extents_uses_fraction_with_minimum():
    """Axis length is a fraction of the largest extent, never below the minimum."""
    assert axis_length_from_extents((1.0, 2.0, 0.5), fraction=0.35, min_m=0.05) == 0.7
    assert axis_length_from_extents((0.01, 0.02, 0.01), fraction=0.35, min_m=0.05) == 0.05


def test_overlay_rgb_segments_on_png_draws_opaque_lines():
    """PNG overlay draws always-on-top segments without needing Kit."""
    import io

    from PIL import Image

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
