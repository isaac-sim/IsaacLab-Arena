# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""World-space XYZ axis math and PNG overlay helpers for review-GUI thumbnails."""

from __future__ import annotations

AXIS_X_COLOR = (1.0, 0.0, 0.0, 1.0)
AXIS_Y_COLOR = (0.0, 1.0, 0.0, 1.0)
AXIS_Z_COLOR = (0.0, 0.0, 1.0, 1.0)


def local_xyz_axis_segments(
    origin_xyz: tuple[float, float, float],
    *,
    x_dir: tuple[float, float, float],
    y_dir: tuple[float, float, float],
    z_dir: tuple[float, float, float],
    length_m: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]]]:
    """Return ``(start, end, rgba)`` segments for local +X/+Y/+Z axes.

    Degenerate directions are omitted. Directions are normalized; ``length_m`` sets
    world-space segment length.
    """
    assert length_m > 0.0, f"length_m must be positive, got {length_m}"
    segments: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]]] = (
        []
    )
    for direction, color in (
        (_unit(x_dir), AXIS_X_COLOR),
        (_unit(y_dir), AXIS_Y_COLOR),
        (_unit(z_dir), AXIS_Z_COLOR),
    ):
        if direction is None:
            continue
        end = (
            origin_xyz[0] + direction[0] * length_m,
            origin_xyz[1] + direction[1] * length_m,
            origin_xyz[2] + direction[2] * length_m,
        )
        segments.append((origin_xyz, end, color))
    return segments


def axis_length_from_extents(
    extents_xyz: tuple[float, float, float], *, fraction: float = 0.35, min_m: float = 0.05
) -> float:
    """Pick a visible axis length from AABB extents (fraction of max side, floored)."""
    assert fraction > 0.0, f"fraction must be positive, got {fraction}"
    assert min_m > 0.0, f"min_m must be positive, got {min_m}"
    return max(min_m, fraction * max(extents_xyz))


def overlay_rgb_segments_on_png(
    png_bytes: bytes,
    segments: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]],
    *,
    width_px: int = 4,
) -> bytes:
    """Draw opaque RGB line segments on top of a PNG and return new PNG bytes."""
    import io

    from PIL import Image, ImageDraw

    with Image.open(io.BytesIO(png_bytes)) as image:
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for start, end, rgb in segments:
            draw.line([start, end], fill=rgb, width=width_px)
        out = io.BytesIO()
        canvas.save(out, format="PNG")
        return out.getvalue()


def _unit(direction: tuple[float, float, float]) -> tuple[float, float, float] | None:
    """Normalize a 3D direction, or None when the vector is degenerate."""
    length = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
    if length < 1e-12:
        return None
    return (direction[0] / length, direction[1] / length, direction[2] / length)
