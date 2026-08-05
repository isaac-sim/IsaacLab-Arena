# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug drawing utilities for Isaac Sim visualization."""

from __future__ import annotations

# Default color for all bounding boxes (bright green)
DEFAULT_COLOR = (0.0, 1.0, 0.0, 1.0)
AXIS_X_COLOR = (1.0, 0.0, 0.0, 1.0)
AXIS_Y_COLOR = (0.0, 1.0, 0.0, 1.0)
AXIS_Z_COLOR = (0.0, 0.0, 1.0, 1.0)


class IsaacSimDebugDraw:
    """Debug drawing utilities for Isaac Sim.

    Wraps Isaac Sim's debug_draw extension to provide convenient methods
    for visualizing bounding boxes, points, and other debug information.

    Note: Debug drawings are overlays rendered on top of the 3D scene.
    They are not part of the USD stage and persist until cleared.

    Example:
        >>> from isaaclab_arena.utils.isaac_sim_debug_draw import IsaacSimDebugDraw
        >>> debug_draw = IsaacSimDebugDraw()
        >>> debug_draw.draw_object_bboxes([cracker_box, office_table])
        >>> debug_draw.clear()
    """

    def __init__(self):
        """Initialize the debug draw interface.

        Automatically enables the isaacsim.util.debug_draw extension if not already enabled.
        """
        self._ensure_extension_enabled()
        from isaacsim.util.debug_draw import _debug_draw

        self._draw = _debug_draw.acquire_debug_draw_interface()

    def _ensure_extension_enabled(self) -> None:
        """Enable the debug_draw extension if not already enabled."""
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)

    def draw_bbox(
        self,
        min_point: tuple[float, float, float],
        max_point: tuple[float, float, float],
        thickness: float = 3.0,
    ) -> None:
        """Draw a single bounding box wireframe from min/max coordinates.

        Args:
            min_point: Minimum corner (x, y, z).
            max_point: Maximum corner (x, y, z).
            thickness: Line thickness in pixels.
        """
        self._draw_bbox_wireframe(min_point, max_point, DEFAULT_COLOR, thickness)

    def draw_object_bboxes(
        self,
        objects: list,
        thickness: float = 3.0,
    ) -> None:
        """Draw bounding boxes for one or more objects.

        Uses each object's get_world_bounding_box() method which returns
        the bounding box in world coordinates (local bbox + position offset).

        Args:
            objects: List of objects with get_world_bounding_box() methods.
            thickness: Line thickness in pixels.
        """
        for obj in objects:
            bbox_coords = self._extract_bbox_from_object(obj)
            if bbox_coords is not None:
                min_pt, max_pt = bbox_coords
                self._draw_bbox_wireframe(min_pt, max_pt, DEFAULT_COLOR, thickness)
            else:
                print(f"Skipping {obj.name}: no bbox coordinates")

    def clear(self) -> None:
        """Clear all debug drawings."""
        self._draw.clear_lines()
        self._draw.clear_points()

    def _extract_bbox_from_object(self, obj) -> tuple[tuple, tuple] | None:
        """Extract world-space bounding box coordinates from an object.

        Returns:
            Tuple of (min_point, max_point) or None if extraction failed.
        """
        world_bbox = obj.get_world_bounding_box()
        return tuple(world_bbox.min_point[0].tolist()), tuple(world_bbox.max_point[0].tolist())

    def _draw_bbox_wireframe(
        self,
        min_point: tuple[float, float, float],
        max_point: tuple[float, float, float],
        color: tuple[float, float, float, float],
        thickness: float,
    ) -> None:
        """Draw a wireframe bounding box using 12 edge lines."""
        x0, y0, z0 = min_point
        x1, y1, z1 = max_point

        # 8 corners of the box
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),  # Bottom face
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),  # Top face
        ]

        # 12 edges connecting corners
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical edges
        ]

        # Build lists for draw_lines API
        start_points = [corners[i] for i, j in edges]
        end_points = [corners[j] for i, j in edges]
        colors_list = [color] * len(edges)
        widths = [thickness] * len(edges)

        self._draw.draw_lines(start_points, end_points, colors_list, widths)


def local_xyz_axis_segments(
    origin_xyz: tuple[float, float, float],
    *,
    x_dir: tuple[float, float, float],
    y_dir: tuple[float, float, float],
    z_dir: tuple[float, float, float],
    length_m: float,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float, float]],
]:
    """Return start/end points and RGBA colors for local +X/+Y/+Z axis lines.

    Args:
        origin_xyz: World-space origin of the axes.
        x_dir: Direction of local +X in world space (normalized internally).
        y_dir: Direction of local +Y in world space (normalized internally).
        z_dir: Direction of local +Z in world space (normalized internally).
        length_m: Axis length in meters.

    Returns:
        ``(start_points, end_points, colors)`` suitable for ``draw_lines``.
    """
    assert length_m > 0.0, f"length_m must be positive, got {length_m}"
    axes = (
        (_unit(x_dir), AXIS_X_COLOR),
        (_unit(y_dir), AXIS_Y_COLOR),
        (_unit(z_dir), AXIS_Z_COLOR),
    )
    start_points: list[tuple[float, float, float]] = []
    end_points: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    for direction, color in axes:
        if direction is None:
            continue
        start_points.append(origin_xyz)
        end_points.append((
            origin_xyz[0] + direction[0] * length_m,
            origin_xyz[1] + direction[1] * length_m,
            origin_xyz[2] + direction[2] * length_m,
        ))
        colors.append(color)
    return start_points, end_points, colors


def axis_length_from_extents(
    extents_xyz: tuple[float, float, float], *, fraction: float = 0.35, min_m: float = 0.05
) -> float:
    """Pick a visible axis length from AABB extents.

    Args:
        extents_xyz: Axis-aligned size ``(x, y, z)`` in meters.
        fraction: Fraction of the largest extent to use.
        min_m: Minimum length so tiny assets still show axes.

    Returns:
        Axis length in meters.
    """
    assert fraction > 0.0, f"fraction must be positive, got {fraction}"
    assert min_m > 0.0, f"min_m must be positive, got {min_m}"
    return max(min_m, fraction * max(extents_xyz))


def overlay_rgb_segments_on_png(
    png_bytes: bytes,
    segments: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]],
    *,
    width_px: int = 4,
) -> bytes:
    """Draw opaque RGB line segments on top of a PNG and return new PNG bytes.

    Args:
        png_bytes: Source snapshot PNG.
        segments: ``(start_uv, end_uv, rgb)`` pixel segments.
        width_px: Stroke width in pixels.

    Returns:
        PNG bytes with the segments composited on top.
    """
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
