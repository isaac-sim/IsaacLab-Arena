# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time cuRobo IK-reachability gate for pooled placement, sim-free (no SimApp).

The pool's solve loop calls it on each geometry-valid candidate; a candidate is stored only when the robot can reach a
top-down grasp at every movable object, so the loop keeps solving (reject-&-refill) until every env has enough reachable layouts.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_events import get_base_rotation_per_object
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.placement_validator_registry import register_validator
from isaaclab_arena.relations.placement_validators import PlacementValidator
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw
from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
from isaaclab_arena_curobo.ik_solver import CuroboIKSolver
from isaaclab_arena_curobo.utils.frame_utils import top_down_grasp_pose_from_world_poses
from isaaclab_arena_curobo.utils.ik_solver_utils import get_aabb_collision_cuboid_for_object, solve_ik_feasibility

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def get_object_world_pose_from_layout(
    positions: dict[ObjectBase, tuple[float, float, float]],
    orientations: dict[ObjectBase, float],
    obj: ObjectBase,
    base_rotations: dict,
) -> Pose:
    """Return the world pose an object gets under a layout."""
    pos_w = positions[obj]
    base_quat_xyzw = base_rotations[obj]
    marker_yaw = yaw_from_quat_xyzw(base_quat_xyzw)
    total_yaw = orientations.get(obj, marker_yaw)
    quat_w_xyzw = rotate_quat_by_yaw(base_quat_xyzw, total_yaw - marker_yaw)
    return Pose(
        position_xyz=tuple(float(v) for v in pos_w),
        rotation_xyzw=tuple(float(v) for v in quat_w_xyzw),
    )


@register_validator
class ReachabilityValidator(PlacementValidator):
    """Build-time placement gate: the robot can reach a top-down grasp at every movable object (cuRobo IK).

    Can be delisted (see ``is_available``) when the params carry no embodiment with a registered cuRobo config.
    """

    check = PlacementCheck.IK_REACHABLE
    run_after_inexpensive_checks = True

    def __init__(self, params: ObjectPlacerParams) -> None:
        super().__init__(params)
        config = params.reachability_config
        self._grasp_z_offset = config.grasp_z_offset_m
        self._ik_pos_threshold = config.ik_position_threshold_m
        self._ik_rot_threshold = config.ik_rotation_threshold_rad
        self._solver = CuroboIKSolver(
            get_embodiment_curobo_cfg(config.embodiment),
            position_threshold=self._ik_pos_threshold,
            rotation_threshold=self._ik_rot_threshold,
        )
        # TODO(xinjieyao, 2026-07-22): Switch to solved pose of the robot base
        base_pose = config.embodiment.get_initial_pose()
        self._base_pos = base_pose.position_xyz
        self._base_quat_xyzw = base_pose.rotation_xyzw

    @classmethod
    def is_available(cls, params: ObjectPlacerParams) -> bool:
        """True when an IK solver can be built for the reachability embodiment (set, with a cuRobo config)."""
        embodiment = params.reachability_config.embodiment
        if embodiment is None:
            return False
        try:
            get_embodiment_curobo_cfg(embodiment)
        except AssertionError:
            # The embodiment has no registered cuRobo config -- treat reachability as unavailable.
            return False
        return True

    def validate_batch(
        self,
        positions: list[dict[ObjectBase, tuple[float, float, float]]],
        orientations: list[dict[ObjectBase, float]],
        bboxes: list[dict[ObjectBase, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], orientations[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        orientations: dict[ObjectBase, float],
    ) -> bool:
        """Whether the robot can reach a top-down grasp at every movable object in one candidate layout.

        Rebuilds each object's world pose and a per-object collision cuboid, syncs them into the solver's
        world, then batches a single IK solve over the movable objects' top-down grasps. An anchor-only
        layout (nothing to grasp) is trivially reachable.
        """
        objects = list(positions.keys())
        anchors = set(get_anchor_objects(objects))
        base_rotations = get_base_rotation_per_object(objects)

        world_poses = {
            obj: get_object_world_pose_from_layout(positions, orientations, obj, base_rotations) for obj in objects
        }
        cuboids = [
            get_aabb_collision_cuboid_for_object(obj, world_poses[obj].position_xyz, world_poses[obj].rotation_xyzw)
            for obj in objects
        ]
        self._solver.update_world(cuboids, self._base_pos, self._base_quat_xyzw)

        movable = [obj for obj in objects if obj not in anchors]
        if not movable:
            return True

        grasp_poses = torch.stack([
            top_down_grasp_pose_from_world_poses(
                world_poses[obj].position_xyz,
                world_poses[obj].rotation_xyzw,
                self._base_pos,
                self._base_quat_xyzw,
                self._grasp_z_offset,
                device=self._solver.device,
            )
            for obj in movable
        ])
        feasible, _, _ = solve_ik_feasibility(
            self._solver,
            grasp_poses,
            position_threshold=self._ik_pos_threshold,
            rotation_threshold=self._ik_rot_threshold,
        )
        return bool(feasible.all().item())
