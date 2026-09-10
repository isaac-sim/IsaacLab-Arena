# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time cuRobo IK-reachability gate for pooled placement, sim-free (no SimApp).

The pool's solve loop calls it on each geometry-valid candidate; a candidate is stored only when the robot can reach a
top-down grasp at every movable object, so the loop keeps solving (reject-&-refill) until every env has enough reachable layouts.

With the placement debug view on (``ObjectPlacerParams.debug_visualize``), the check also draws what it solved for each
candidate -- the robot base, the grasps, their IK errors; see ``isaaclab_arena_curobo.reachability_visualizer``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_events import get_base_rotation_per_asset
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.placement_validator_registry import register_validator
from isaaclab_arena.relations.placement_validators import PlacementValidator
from isaaclab_arena.relations.relations import RequiresReachability, get_anchor_objects
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw
from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
from isaaclab_arena_curobo.ik_solver import CuroboIKSolver
from isaaclab_arena_curobo.utils.frame_utils import top_down_grasp_pose_from_world_poses
from isaaclab_arena_curobo.utils.ik_solver_utils import (
    AABBCollisionCuboid,
    IKFeasibility,
    get_aabb_collision_cuboid_for_object,
    hand_sphere_mask,
    robot_collision_spheres,
    solve_ik_feasibility,
)

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena_curobo.reachability_visualizer import ReachabilityRerunLayer


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
    """Build-time placement gate: the robot can reach a top-down grasp at the target objects (cuRobo IK).
    Can be delisted (see ``is_available``) when the params carry no embodiment with a registered cuRobo config.
    """

    check = PlacementCheck.IK_REACHABLE
    run_after_inexpensive_checks = True

    def __init__(self, params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None) -> None:
        super().__init__(params, visualizer)
        config = params.reachability_config
        self._grasp_z_offset = config.grasp_z_offset_m
        self._ik_pos_threshold = config.ik_position_threshold_m
        self._ik_rot_threshold = config.ik_rotation_threshold_rad
        self._require_collision_free = config.require_collision_free
        self._solver = CuroboIKSolver(
            get_embodiment_curobo_cfg(config.embodiment),
            position_threshold=self._ik_pos_threshold,
            rotation_threshold=self._ik_rot_threshold,
        )
        self._embodiment = config.embodiment
        # Used only when the robot is not itself relation-placed, in which case it never appears in a layout.
        self._configured_robot_base_pose_w = config.embodiment.get_initial_pose()
        # Guards the zero-target warning so it fires once per validator, not once per candidate layout.
        self._warned_no_targets = False
        self._rerun_layer = self._make_rerun_layer()

    def _make_rerun_layer(self) -> ReachabilityRerunLayer | None:
        """Return this check's layer of the placement visualizer, or None when no one asked for it."""
        if self._visualizer is None:
            return None
        from isaaclab_arena_curobo.reachability_visualizer import ReachabilityRerunLayer

        return ReachabilityRerunLayer(self._visualizer)

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
        return [
            self._validate(positions[i], orientations[i], layout_index_within_batch=i) for i in range(len(positions))
        ]

    def _validate(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        orientations: dict[ObjectBase, float],
        layout_index_within_batch: int,
    ) -> bool:
        """Whether the robot can reach a top-down grasp at the target objects in one candidate layout.

        Rebuilds each object's world pose, builds a collision cuboid per object -- what makes them block a
        grasp when collision checking is on -- then IK-solves each target's top-down grasp against a world
        holding every other object. A layout with nothing to grasp (anchor-only, or no target present) is
        trivially reachable. The robot stands where this layout puts it when it is relation-placed too,
        so the grasps are solved from the base pose the layout would actually spawn.

        Args:
            positions: Solved (x, y, z) per object.
            orientations: Absolute world Z-yaw per object.
            layout_index_within_batch: Position of this layout in the batch given to ``validate_batch``.
        """
        objects = list(positions.keys())
        anchors = set(get_anchor_objects(objects))
        base_rotations = get_base_rotation_per_asset(objects)

        world_poses = {
            obj: get_object_world_pose_from_layout(positions, orientations, obj, base_rotations) for obj in objects
        }
        # non-anchor objects with a RequiresReachability relation
        targets = self._select_reachability_targets(objects, anchors)
        robot_base_pose_w = world_poses.get(self._embodiment, self._configured_robot_base_pose_w)
        # The robot's own body is not an obstacle: cuRobo already carries it as collision spheres.
        cuboid_per_object = {
            obj: get_aabb_collision_cuboid_for_object(
                obj, world_poses[obj].position_xyz, world_poses[obj].rotation_xyzw
            )
            for obj in objects
            if obj is not self._embodiment
        }

        if not targets:
            # The check is enabled but no movable object is stamped as a reachability target, so it passes every
            # layout trivially.
            if not self._warned_no_targets:
                print(
                    "[ReachabilityValidator] WARNING: enabled but resolved zero reachability targets; every layout "
                    "passes the IK check trivially. No reachability targets found in the task."
                )
                self._warned_no_targets = True
            return True

        grasp_poses = torch.stack([
            top_down_grasp_pose_from_world_poses(
                world_poses[obj].position_xyz,
                world_poses[obj].rotation_xyzw,
                robot_base_pose_w.position_xyz,
                robot_base_pose_w.rotation_xyzw,
                self._grasp_z_offset,
                device=self._solver.device,
            )
            for obj in targets
        ])
        ik = self._solve_grasp_per_target(targets, grasp_poses, cuboid_per_object, robot_base_pose_w)
        if self._rerun_layer is not None:
            layout_index_across_batch = self._visualizer.get_layout_index_across_batch(layout_index_within_batch)
            self._rerun_layer.log_layout(
                layout_index_across_batch=layout_index_across_batch,
                robot_base_pos_w=robot_base_pose_w.position_xyz,
                robot_base_quat_w_xyzw=robot_base_pose_w.rotation_xyzw,
                target_names=[obj.name for obj in targets],
                grasp_poses_base_frame=grasp_poses,
                feasible=ik.feasible,
                position_error=ik.position_error,
                rotation_error=ik.rotation_error,
                # The pose the arm ended up in is what explains a rejection, so it is drawn alongside it.
                robot_spheres=robot_collision_spheres(self._solver, ik.joint_positions),
                muted_sphere_mask=hand_sphere_mask(self._solver) if self._require_collision_free else None,
            )
        return bool(ik.feasible.all().item())

    def _solve_grasp_per_target(
        self,
        targets: list[ObjectBase],
        grasp_poses: torch.Tensor,
        cuboid_per_object: dict[ObjectBase, AABBCollisionCuboid],
        robot_base_pose_w: Pose,
    ) -> IKFeasibility:
        """IK-solve each target's grasp against a world holding every object but that target, and stack the results.

        Args:
            targets: The objects whose grasps are checked, in the order they appear in ``grasp_poses``.
            grasp_poses: ``(len(targets), 4, 4)`` top-down grasp transforms in the robot base frame.
            cuboid_per_object: Collision cuboid at the layout pose, for every object in the layout.
            robot_base_pose_w: World pose of the robot base the obstacles are re-expressed relative to.

        Returns:
            One ``IKFeasibility`` whose per-target entries are ordered like ``targets``.
        """
        per_target = []
        for target_index, target in enumerate(targets):
            obstacles = [cuboid for obj, cuboid in cuboid_per_object.items() if obj is not target]
            self._solver.update_world(obstacles, robot_base_pose_w.position_xyz, robot_base_pose_w.rotation_xyzw)
            per_target.append(
                solve_ik_feasibility(
                    self._solver,
                    grasp_poses[target_index : target_index + 1],
                    position_threshold=self._ik_pos_threshold,
                    rotation_threshold=self._ik_rot_threshold,
                    require_collision_free=self._require_collision_free,
                )
            )
        return IKFeasibility(
            feasible=torch.cat([ik.feasible for ik in per_target]),
            position_error=torch.cat([ik.position_error for ik in per_target]),
            rotation_error=torch.cat([ik.rotation_error for ik in per_target]),
            joint_positions=torch.cat([ik.joint_positions for ik in per_target]),
        )

    def _select_reachability_targets(self, objects: list[ObjectBase], anchors: set[ObjectBase]) -> list[ObjectBase]:
        """Movable objects the task marked as reachability targets (carry a RequiresReachability relation)."""
        return [obj for obj in objects if obj not in anchors and obj.has_relation(RequiresReachability)]
