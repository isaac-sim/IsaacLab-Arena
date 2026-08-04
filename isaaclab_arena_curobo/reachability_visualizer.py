# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The reachability check's layer of the placement Rerun debug view, sim-free (no SimApp).
Adds to existing visualizer what only the IK check knows -- where the
robot stands, the top-down grasps it solved, and whether each one was reachable.
"""

from __future__ import annotations

import torch

from isaaclab_arena.relations.placement_visualizer import ROBOT_ENTITY, PlacementRerunVisualizer

REACHABLE_COLOR = (40, 200, 80)
"""Color of a grasp the robot can reach."""

UNREACHABLE_COLOR = (220, 50, 50)
"""Color of a grasp the robot cannot reach, i.e. the one that rejected the layout."""

BASE_AXIS_LENGTH = 0.2
"""Length (m) of the drawn robot base frame axes."""

GRASP_AXIS_LENGTH = 0.1
"""Length (m) of the drawn grasp frame axes."""

BASE_ENTITY = f"{ROBOT_ENTITY}/base"
"""Entity path of the robot base frame; the grasps below it are logged in that frame."""


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
        """
        import rerun as rr

        self._visualizer.set_time(layout_index_across_batch)
        # Grasps are solved in the robot base frame, so they are logged as children of the base
        # transform and Rerun composes them back into the world frame.
        rr.log(
            BASE_ENTITY,
            rr.Transform3D(translation=robot_base_pos_w, quaternion=rr.Quaternion(xyzw=robot_base_quat_w_xyzw)),
        )
        rr.log(BASE_ENTITY, rr.TransformAxes3D(BASE_AXIS_LENGTH))

        grasps = grasp_poses_base_frame.detach().cpu()
        for i, name in enumerate(target_names):
            reachable = bool(feasible[i].item())
            color = REACHABLE_COLOR if reachable else UNREACHABLE_COLOR
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
