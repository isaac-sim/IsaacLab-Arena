# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug drawing utilities for Isaac Sim visualization."""

from __future__ import annotations

import math
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh

DEFAULT_COLOR = (0.0, 1.0, 0.0, 1.0)

_BBOX_EDGE_INDICES = (
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
)


def oriented_bbox_corners(
    min_point: tuple[float, float, float] | np.ndarray,
    max_point: tuple[float, float, float] | np.ndarray,
    position_xyz: tuple[float, float, float] | np.ndarray,
    rotation_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return the 8 world-space corners of a local AABB under a rigid pose.

    Args:
        min_point: Local AABB minimum corner.
        max_point: Local AABB maximum corner.
        position_xyz: World translation of the local origin.
        rotation_xyzw: World orientation as ``(x, y, z, w)``.

    Returns:
        Array of shape ``(8, 3)`` with corner positions in world coordinates.
    """
    x0, y0, z0 = (float(v) for v in min_point)
    x1, y1, z1 = (float(v) for v in max_point)
    local = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    return rotate_points_xyzw(local, rotation_xyzw) + np.asarray(position_xyz, dtype=np.float64)


def oriented_bbox_edge_segments(
    min_point: tuple[float, float, float] | np.ndarray,
    max_point: tuple[float, float, float] | np.ndarray,
    position_xyz: tuple[float, float, float] | np.ndarray,
    rotation_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return start/end point lists for the 12 edges of an oriented AABB."""
    corners = oriented_bbox_corners(min_point, max_point, position_xyz, rotation_xyzw)
    starts = [tuple(corners[i].tolist()) for i, _ in _BBOX_EDGE_INDICES]
    ends = [tuple(corners[j].tolist()) for _, j in _BBOX_EDGE_INDICES]
    return starts, ends


def rotate_points_xyzw(
    points: np.ndarray,
    rotation_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Rotate ``(N, 3)`` points by a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = (float(v) for v in rotation_xyzw)
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, points)
    uuv = np.cross(qvec, uv)
    return points + 2.0 * (w * uv + uuv)


def trimesh_edge_segments(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edges: int | None = 8000,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Build unique mesh edges as start/end point lists, optionally stride-decimated.

    Args:
        vertices: Mesh vertices, shape ``(V, 3)``.
        faces: Triangle faces, shape ``(F, 3)``.
        max_edges: Maximum number of edges to keep; stride-sample when over budget.
            Set to ``None`` to keep every edge.

    Returns:
        Parallel lists of edge start and end points.
    """
    if len(faces) == 0 or len(vertices) == 0:
        return [], []
    edges: set[tuple[int, int]] = set()
    for face in faces:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        edges.add((min(i, j), max(i, j)))
        edges.add((min(j, k), max(j, k)))
        edges.add((min(k, i), max(k, i)))
    edge_list = sorted(edges)
    if max_edges is not None and len(edge_list) > max_edges:
        stride = math.ceil(len(edge_list) / max_edges)
        edge_list = edge_list[::stride][:max_edges]
    verts = np.asarray(vertices, dtype=np.float64)
    starts = [tuple(verts[i].tolist()) for i, _ in edge_list]
    ends = [tuple(verts[j].tolist()) for _, j in edge_list]
    return starts, ends


def transform_trimesh_vertices(
    vertices: np.ndarray,
    position_xyz: tuple[float, float, float] | np.ndarray,
    rotation_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Apply a rigid pose to mesh vertices (local → world)."""
    return rotate_points_xyzw(np.asarray(vertices, dtype=np.float64), rotation_xyzw) + np.asarray(
        position_xyz, dtype=np.float64
    )


class IsaacSimDebugDraw:
    """Debug drawing utilities for Isaac Sim.

    Wraps Isaac Sim's debug_draw extension to provide convenient methods
    for visualizing bounding boxes, points, and other debug information.

    Note: Debug drawings are overlays rendered on top of the 3D scene.
    They are not part of the USD stage and persist until cleared.

    Example:
        >>> from isaaclab_arena.visualization.isaac_sim_debug_draw import IsaacSimDebugDraw
        >>> debug_draw = IsaacSimDebugDraw()
        >>> debug_draw.draw_object_bboxes([cracker_box, office_table])
        >>> debug_draw.clear()
    """

    def __init__(self):
        """Initialize the debug draw interface.

        Automatically enables the isaacsim.util.debug_draw extension if not already enabled.
        """
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
        from isaacsim.util.debug_draw import _debug_draw

        self._draw = _debug_draw.acquire_debug_draw_interface()

    def draw_lines(
        self,
        start_points: list[tuple[float, float, float]],
        end_points: list[tuple[float, float, float]],
        color: tuple[float, float, float, float] = DEFAULT_COLOR,
        thickness: float = 3.0,
    ) -> None:
        """Draw a batch of line segments in a single color."""
        if not start_points:
            return
        assert len(start_points) == len(end_points), "start_points and end_points must have equal length"
        colors_list = [color] * len(start_points)
        widths = [thickness] * len(start_points)
        self._draw.draw_lines(start_points, end_points, colors_list, widths)

    def draw_bbox(
        self,
        min_point: tuple[float, float, float],
        max_point: tuple[float, float, float],
        thickness: float = 3.0,
        color: tuple[float, float, float, float] = DEFAULT_COLOR,
    ) -> None:
        """Draw a single axis-aligned bounding box wireframe from min/max coordinates.

        Args:
            min_point: Minimum corner (x, y, z).
            max_point: Maximum corner (x, y, z).
            thickness: Line thickness in pixels.
            color: RGBA color in ``[0, 1]``.
        """
        self.draw_oriented_bbox(
            min_point,
            max_point,
            position_xyz=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            color=color,
            thickness=thickness,
        )

    def draw_oriented_bbox(
        self,
        min_point: tuple[float, float, float],
        max_point: tuple[float, float, float],
        position_xyz: tuple[float, float, float],
        rotation_xyzw: tuple[float, float, float, float],
        color: tuple[float, float, float, float] = DEFAULT_COLOR,
        thickness: float = 3.0,
    ) -> None:
        """Draw a local AABB transformed by a live rigid pose."""
        starts, ends = oriented_bbox_edge_segments(min_point, max_point, position_xyz, rotation_xyzw)
        self.draw_lines(starts, ends, color=color, thickness=thickness)

    def draw_trimesh_wireframe(
        self,
        mesh: trimesh.Trimesh,
        color: tuple[float, float, float, float] = DEFAULT_COLOR,
        thickness: float = 1.5,
        max_edges: int = 8000,
        position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        """Draw mesh edges as a wireframe, optionally under a rigid pose.

        Large meshes are stride-decimated to at most ``max_edges`` segments.
        """
        vertices = transform_trimesh_vertices(mesh.vertices, position_xyz, rotation_xyzw)
        starts, ends = trimesh_edge_segments(vertices, np.asarray(mesh.faces), max_edges=max_edges)
        self.draw_lines(starts, ends, color=color, thickness=thickness)

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
            world_bbox = obj.get_world_bounding_box()
            self.draw_bbox(
                tuple(world_bbox.min_point[0].tolist()),
                tuple(world_bbox.max_point[0].tolist()),
                thickness=thickness,
            )

    def clear(self) -> None:
        """Clear all debug drawings."""
        self._draw.clear_lines()
        self._draw.clear_points()
