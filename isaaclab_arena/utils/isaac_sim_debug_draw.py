# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug drawing utilities for Isaac Sim visualization."""

# Default color for all bounding boxes (bright green)
DEFAULT_COLOR = (0.0, 1.0, 0.0, 1.0)


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
        x0, y0, z0 = min_point
        x1, y1, z1 = max_point
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        self._draw_corner_wireframe(corners, DEFAULT_COLOR, thickness)

    def draw_object_bboxes(
        self,
        objects: list,
        thickness: float = 3.0,
    ) -> None:
        """Draw bounding boxes for one or more objects.

        Uses each object's exact oriented world bounding box.

        Args:
            objects: List of objects with get_world_bounding_box() methods.
            thickness: Line thickness in pixels.
        """
        for obj in objects:
            corners = obj.get_world_bounding_box().get_corners()[0].tolist()
            self._draw_corner_wireframe(corners, DEFAULT_COLOR, thickness)

    def clear(self) -> None:
        """Clear all debug drawings."""
        self._draw.clear_lines()
        self._draw.clear_points()

    def _draw_corner_wireframe(
        self,
        corners: list[tuple[float, float, float]] | list[list[float]],
        color: tuple[float, float, float, float],
        thickness: float,
    ) -> None:
        """Draw the 12-edge wireframe joining eight ordered corners."""
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

        start_points = [corners[i] for i, j in edges]
        end_points = [corners[j] for i, j in edges]
        colors_list = [color] * len(edges)
        widths = [thickness] * len(edges)

        self._draw.draw_lines(start_points, end_points, colors_list, widths)
