# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The reachability check's layer of the placement Rerun debug view, sim-free (no SimApp).
Adds to existing visualizer what only the IK check knows -- where the
robot stands, the top-down grasps it solved, whether each one was reachable, the collision
spheres the robot holds at each solved grasp, and the background mesh it solved against, with the
triangles the robot hits or nearly hits highlighted.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.validation.placement_visualizer import ROBOT_ENTITY, PlacementRerunVisualizer

if TYPE_CHECKING:
    import trimesh

REACHABLE_COLOR = (40, 200, 80)
"""Color of a grasp the robot can reach."""

UNREACHABLE_COLOR = (220, 50, 50)
"""Color of a grasp the robot cannot reach, i.e. the one that rejected the layout."""

BASE_AXIS_LENGTH = 0.2
"""Length (m) of the drawn robot base frame axes."""

GRASP_AXIS_LENGTH = 0.1
"""Length (m) of the drawn grasp frame axes."""

MUTED_SPHERE_COLOR = (235, 180, 60)
"""Color of a collision sphere muted for the solve, i.e. one allowed to touch the layout."""

SPHERE_ALPHA = 110
"""Opacity of the drawn collision spheres, so the layout stays visible through the robot."""

BASE_ENTITY = f"{ROBOT_ENTITY}/base"
"""Entity path of the robot base frame; the grasps below it are logged in that frame."""

SPHERES_ENTITY = f"{BASE_ENTITY}/collision_spheres"
"""Entity path of the robot's collision spheres, one child per grasp the check solved."""

LOADED_BACKGROUND_ENTITY = f"{ROBOT_ENTITY}/background_obstacles"
"""Entity path of the background mesh loaded for one layout; under the robot so a skipped layout clears it."""

LOADED_BACKGROUND_COLOR = (160, 170, 190, 70)
"""Color of the background mesh loaded as an obstacle for the layout's solves."""

NEAR_BACKGROUND_MARGIN_M = 0.05
"""Gap (m) between a background triangle and a collision-checked sphere below which the triangle is highlighted."""

NEAR_BACKGROUND_COLOR = (230, 140, 40)
"""Color of a background triangle within the margin of the robot, without touching it."""

HIT_BACKGROUND_COLOR = (220, 50, 50)
"""Color of a background triangle a collision-checked sphere penetrates."""


class ReachabilityRerunLayer:
    """Draws the reachability check's verdict for a layout into the shared placement view."""

    def __init__(self, visualizer: PlacementRerunVisualizer) -> None:
        """Bind the layer to the placement visualizer.

        Args:
            visualizer: The process's placement visualizer, which owns the recording and the timeline.
        """
        self._visualizer = visualizer

    def log_layout(
        self,
        layout_index_across_batch: int,
        robot_base_pos_w: tuple[float, float, float],
        robot_base_quat_w_xyzw: tuple[float, float, float, float],
        target_names: list[str],
        grasp_poses_base_frame: torch.Tensor,
        feasible: torch.Tensor,
        position_error: torch.Tensor,
        rotation_error: torch.Tensor,
        robot_spheres: torch.Tensor | None = None,
        muted_sphere_mask: torch.Tensor | None = None,
        background_mesh_W: trimesh.Trimesh | None = None,
    ) -> None:
        """Log the robot's side of one evaluated layout.

        Args:
            layout_index_across_batch: Timeline index of the layout, as assigned by the placement view.
            robot_base_pos_w: Robot base frame position in the world frame.
            robot_base_quat_w_xyzw: Robot base frame orientation in the world frame.
            target_names: Names of the objects a grasp was solved for, aligned with the tensors below.
            grasp_poses_base_frame: ``(b, 4, 4)`` grasp transforms in the robot base frame.
            feasible: ``(b,)`` per-grasp IK verdict.
            position_error: ``(b,)`` per-grasp IK position error (m).
            rotation_error: ``(b,)`` per-grasp IK rotation error (rad).
            robot_spheres: ``(b, n, 4)`` collision spheres ``(x, y, z, radius)`` in the robot base frame,
                one set per grasp; omitted when the check did not compute them.
            muted_sphere_mask: ``(n,)`` mask of the spheres excluded from collision checking, drawn
                apart from the rest. None means none were.
            background_mesh_W: Background mesh loaded as an obstacle for this layout's solves, in the world
                frame; omitted when the background is not loaded.
        """
        import rerun as rr

        self._visualizer.set_time(layout_index_across_batch)
        if background_mesh_W is not None and len(background_mesh_W.faces) > 0:
            self._log_background(
                background_mesh_W, robot_base_pos_w, robot_base_quat_w_xyzw, robot_spheres, muted_sphere_mask
            )
        # Grasps are solved in the robot base frame, so they are logged as children of the base
        # transform and Rerun composes them back into the world frame.
        rr.log(
            BASE_ENTITY,
            rr.Transform3D(translation=robot_base_pos_w, quaternion=rr.Quaternion(xyzw=robot_base_quat_w_xyzw)),
        )
        rr.log(BASE_ENTITY, rr.TransformAxes3D(BASE_AXIS_LENGTH))

        grasps = grasp_poses_base_frame.detach().cpu()
        spheres = None if robot_spheres is None else robot_spheres.detach().cpu()
        muted = None if muted_sphere_mask is None else muted_sphere_mask.detach().cpu()
        for i, name in enumerate(target_names):
            reachable = bool(feasible[i].item())
            color = REACHABLE_COLOR if reachable else UNREACHABLE_COLOR
            if spheres is not None:
                self._log_robot_spheres(f"{SPHERES_ENTITY}/{name}", spheres[i], muted, color)
            entity = f"{BASE_ENTITY}/grasps/{name}"
            rr.log(entity, rr.Transform3D(translation=grasps[i, :3, 3], mat3x3=grasps[i, :3, :3]))
            rr.log(entity, rr.TransformAxes3D(GRASP_AXIS_LENGTH))
            rr.log(
                f"{entity}/verdict",
                rr.Points3D(
                    [[0.0, 0.0, 0.0]],
                    colors=[color],
                    radii=0.015,
                    labels=[f"{name}: {'reachable' if reachable else 'unreachable'}"],
                ),
            )
            rr.log(f"errors/{name}/position_m", rr.Scalars(float(position_error[i].item())))
            rr.log(f"errors/{name}/rotation_rad", rr.Scalars(float(rotation_error[i].item())))

    @staticmethod
    def _log_robot_spheres(
        entity: str,
        spheres: torch.Tensor,
        muted_sphere_mask: torch.Tensor | None,
        checked_color: tuple[int, int, int],
    ) -> None:
        """Draw the robot's collision spheres at one solved grasp.

        Args:
            entity: Entity path to draw under; a child of the base transform, as the spheres are in that frame.
            spheres: ``(n, 4)`` spheres ``(x, y, z, radius)`` in the robot base frame.
            muted_sphere_mask: ``(n,)`` mask of the spheres excluded from collision checking, or None for none.
            checked_color: Color of the collision-checked spheres; the grasp's own verdict color.
        """
        import rerun as rr

        radii = spheres[:, 3]
        # cuRobo leaves spheres it does not use in the tensor with a negative radius, which Rerun would
        # read as a radius in screen points rather than as metres.
        kept = radii > 0.0
        muted = torch.zeros_like(kept) if muted_sphere_mask is None else muted_sphere_mask
        rr.log(
            entity,
            rr.Points3D(
                spheres[kept][:, :3],
                radii=radii[kept],
                colors=[
                    (*MUTED_SPHERE_COLOR, SPHERE_ALPHA) if is_muted else (*checked_color, SPHERE_ALPHA)
                    for is_muted in muted[kept].tolist()
                ],
            ),
        )

    @staticmethod
    def _log_background(
        mesh_W: trimesh.Trimesh,
        robot_base_pos_w: tuple[float, float, float],
        robot_base_quat_w_xyzw: tuple[float, float, float, float],
        robot_spheres: torch.Tensor | None,
        muted_sphere_mask: torch.Tensor | None,
    ) -> None:
        """Draw the loaded background mesh see-through and highlight the triangles the robot hits or nearly hits.

        Args:
            mesh_W: Background mesh loaded for the layout, in the world frame.
            robot_base_pos_w: Robot base frame position in the world frame.
            robot_base_quat_w_xyzw: Robot base frame orientation in the world frame.
            robot_spheres: ``(b, n, 4)`` collision spheres in the robot base frame, one set per grasp, or None.
            muted_sphere_mask: ``(n,)`` mask of the spheres excluded from collision checking, or None for none.
        """
        import rerun as rr

        _log_triangles(f"{LOADED_BACKGROUND_ENTITY}/surface", mesh_W, LOADED_BACKGROUND_COLOR)
        if robot_spheres is not None:
            spheres_W = _checked_spheres_in_world(
                robot_spheres, muted_sphere_mask, robot_base_pos_w, robot_base_quat_w_xyzw
            )
            triangle_ids, gaps = _closest_triangle_gaps(mesh_W, spheres_W)
            highlights = (
                ("hit", gaps < 0.0, HIT_BACKGROUND_COLOR),
                ("near", (gaps >= 0.0) & (gaps < NEAR_BACKGROUND_MARGIN_M), NEAR_BACKGROUND_COLOR),
            )
            for name, selected, color in highlights:
                entity = f"{LOADED_BACKGROUND_ENTITY}/{name}"
                if selected.any():
                    faces = np.unique(triangle_ids[selected])
                    _log_triangles(entity, mesh_W.submesh([faces], append=True), color)
                else:
                    # Otherwise the previous layout's highlight would still show on this one.
                    rr.log(entity, rr.Clear(recursive=False))


def _log_triangles(entity: str, mesh_W: trimesh.Trimesh, color: tuple[int, ...]) -> None:
    """Log a world-frame mesh flat-shaded in one color."""
    import rerun as rr

    flat_W = mesh_W.copy()
    # Split the vertices per face so each face is shaded by its own normal, not smoothed across sharp edges.
    flat_W.unmerge_vertices()
    rr.log(
        entity,
        rr.Mesh3D(
            vertex_positions=flat_W.vertices,
            triangle_indices=flat_W.faces,
            vertex_normals=flat_W.vertex_normals,
            albedo_factor=color,
        ),
    )


def _checked_spheres_in_world(
    robot_spheres: torch.Tensor,
    muted_sphere_mask: torch.Tensor | None,
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """Return the collision-checked spheres of every grasp as ``(k, 4)`` world-frame ``(x, y, z, radius)``.

    Muted spheres and the ones cuRobo leaves unused (negative radius) are dropped.
    """
    import isaaclab.utils.math as math_utils

    spheres_B = robot_spheres.detach().cpu().double()
    checked = spheres_B[..., 3] > 0.0
    if muted_sphere_mask is not None:
        checked &= ~muted_sphere_mask.detach().cpu()
    spheres_B = spheres_B[checked]
    R_W_B = math_utils.matrix_from_quat(torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float64))
    t_W = spheres_B[:, :3] @ R_W_B.T + torch.tensor(robot_base_pos_w, dtype=torch.float64)
    return torch.cat([t_W, spheres_B[:, 3:]], dim=1).numpy()


def _closest_triangle_gaps(mesh_W: trimesh.Trimesh, spheres_W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, per sphere, the mesh triangle closest to it and the gap (m) to it; negative where the sphere penetrates.

    Args:
        mesh_W: Mesh in the world frame.
        spheres_W: ``(k, 4)`` world-frame spheres ``(x, y, z, radius)``.

    Returns:
        ``(k,)`` closest triangle indices and ``(k,)`` gaps between the sphere surface and that triangle.
    """
    import trimesh

    if len(spheres_W) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    _, distances, triangle_ids = trimesh.proximity.closest_point(mesh_W, spheres_W[:, :3])
    return np.asarray(triangle_ids), distances - spheres_W[:, 3]
