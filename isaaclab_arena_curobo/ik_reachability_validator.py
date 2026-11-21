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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.optimization.placement_candidate_batch import PlacementCandidateBatch
from isaaclab_arena.relations.placement_diagnostics import Rejection
from isaaclab_arena.relations.relations import RequiresReachability, get_anchor_objects
from isaaclab_arena.relations.runtime.placement_events import get_base_rotation_per_asset
from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator
from isaaclab_arena.relations.validation.registry import register_validator
from isaaclab_arena.relations.validation.types import PlacementCheck
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
    from isaaclab_arena.relations.geometry.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.validation.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena_curobo.reachability_visualizer import ReachabilityRerunLayer

# Maximum number of grasps a cuRobo solve can hold.
IK_MAX_BATCH_SIZE = 256


@dataclass
class CandidateLayoutGrasps:
    """The grasps to check in one candidate layout, and the object poses they are checked against.

    A candidate layout is one proposed arrangement of all the objects in one environment.
    """

    targets: list[ObjectBase]
    """Objects whose top-down grasp is checked."""
    grasp_poses: torch.Tensor
    """``(len(targets), 4, 4)`` top-down grasp transforms in the robot base frame."""
    cuboid_per_object: dict[ObjectBase, AABBCollisionCuboid]
    """Collision cuboid of every object in the layout but the robot, at its world pose in the layout."""
    robot_base_pose_w: Pose
    """World pose of the robot base."""


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
class ReachabilityValidator(PrePhysicsPlacementValidator):
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
            max_batch_size=IK_MAX_BATCH_SIZE,
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

    def _validate_batch(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], rejections: list[Rejection]
    ) -> list[bool]:
        # Batching: instead of one cuRobo call per candidate layout, first collect the grasps of all candidate
        # layouts, then solve them together (each in its own collision world), and only afterwards split the
        # results back into one verdict per layout.
        candidate_grasps = [
            self._get_candidate_grasps_for_layout(candidate.positions, candidate.orientations)
            for candidate in batch.candidates
        ]
        ik_per_layout = self._solve_candidate_grasps(candidate_grasps)
        verdict_per_layout = []
        for layout_index_within_batch, (layout_grasps, ik) in enumerate(
            zip(candidate_grasps, ik_per_layout, strict=True)
        ):
            if layout_grasps is None:
                # No movable object is a reachability target, so the layout is trivially reachable.
                verdict_per_layout.append(True)
                continue
            self._visualize_layout(layout_grasps, ik, layout_index_within_batch)
            self._record_failure(rejections, layout_grasps.targets, ik)
            # A layout is reachable only if every target's grasp is.
            verdict_per_layout.append(bool(ik.feasible.all().item()))
        return verdict_per_layout

    def _get_candidate_grasps_for_layout(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        orientations: dict[ObjectBase, float],
    ) -> CandidateLayoutGrasps | None:
        """Return the grasps to check in one candidate layout, or None when it has nothing to grasp.

        Rebuilds each object's world pose and builds a collision cuboid per object -- what makes them block a
        grasp when collision checking is on. The robot stands where this layout puts it when it is
        relation-placed too, so the grasps are solved from the base pose the layout would actually spawn.

        Args:
            positions: Solved (x, y, z) per object.
            orientations: Absolute world Z-yaw per object.
        """
        objects = list(positions.keys())
        anchors = set(get_anchor_objects(objects))
        base_rotations = get_base_rotation_per_asset(objects)

        world_poses = {
            obj: get_object_world_pose_from_layout(positions, orientations, obj, base_rotations) for obj in objects
        }
        # non-anchor objects with a RequiresReachability relation
        targets = self._select_reachability_targets(objects, anchors)
        if not targets:
            # The check is enabled but no movable object is stamped as a reachability target, so it passes every
            # layout trivially.
            if not self._warned_no_targets:
                print(
                    "[ReachabilityValidator] WARNING: enabled but resolved zero reachability targets; every layout "
                    "passes the IK check trivially. No reachability targets found in the task."
                )
                self._warned_no_targets = True
            return None
        robot_base_pose_w = world_poses.get(self._embodiment, self._configured_robot_base_pose_w)
        # The robot's own body is not an obstacle: cuRobo already carries it as collision spheres.
        cuboid_per_object = {
            obj: get_aabb_collision_cuboid_for_object(
                obj, world_poses[obj].position_xyz, world_poses[obj].rotation_xyzw
            )
            for obj in objects
            if obj is not self._embodiment
        }
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
        return CandidateLayoutGrasps(targets, grasp_poses, cuboid_per_object, robot_base_pose_w)

    def _solve_candidate_grasps(
        self, candidate_grasps: list[CandidateLayoutGrasps | None]
    ) -> list[IKFeasibility | None]:
        """IK-solve every target grasp of every candidate layout, each against a world holding every object but that target.

        The grasps are solved in chunks of ``IK_MAX_BATCH_SIZE``; each has its own collision world.

        Args:
            candidate_grasps: Per candidate layout, its grasps, or None when it has nothing to grasp.

        Returns:
            Per candidate layout, an ``IKFeasibility`` whose entries are ordered like that layout's targets; None
            where the layout has nothing to grasp.
        """
        # Flatten to one IK problem per (layout, target): the target's grasp pose, the robot base pose, and the
        # obstacles of that layout (every object except the target itself, which the gripper must touch).
        problem_layout_indices: list[int] = []
        problem_obstacles: list[list[AABBCollisionCuboid]] = []
        problem_base_poses: list[Pose] = []
        problem_grasp_poses: list[torch.Tensor] = []
        for layout_index, layout_grasps in enumerate(candidate_grasps):
            if layout_grasps is None:
                continue
            for target_index, target in enumerate(layout_grasps.targets):
                problem_layout_indices.append(layout_index)
                problem_obstacles.append(
                    [cuboid for obj, cuboid in layout_grasps.cuboid_per_object.items() if obj is not target]
                )
                problem_base_poses.append(layout_grasps.robot_base_pose_w)
                problem_grasp_poses.append(layout_grasps.grasp_poses[target_index])

        # Solve the problems in chunks that fit one cuRobo solve: load each problem's own collision world, then
        # solve all the chunk's grasps at once.
        results: list[IKFeasibility] = []
        for start in range(0, len(problem_layout_indices), IK_MAX_BATCH_SIZE):
            chunk = slice(start, start + IK_MAX_BATCH_SIZE)
            base_poses = problem_base_poses[chunk]
            self._solver.load_collision_worlds(
                problem_obstacles[chunk],
                [pose.position_xyz for pose in base_poses],
                [pose.rotation_xyzw for pose in base_poses],
            )
            results.append(
                solve_ik_feasibility(
                    self._solver,
                    torch.stack(problem_grasp_poses[chunk]),
                    position_threshold=self._ik_pos_threshold,
                    rotation_threshold=self._ik_rot_threshold,
                    require_collision_free=self._require_collision_free,
                )
            )
        if not results:
            return [None] * len(candidate_grasps)
        # Join the per-chunk results into one entry per problem, in the order the problems were flattened.
        feasible = torch.cat([ik.feasible for ik in results])
        position_error = torch.cat([ik.position_error for ik in results])
        rotation_error = torch.cat([ik.rotation_error for ik in results])
        joint_positions = torch.cat([ik.joint_positions for ik in results])

        # Split the per-problem results back into one IKFeasibility per layout: each layout owns the consecutive
        # problems of its targets.
        ik_per_layout: list[IKFeasibility | None] = [None] * len(candidate_grasps)
        next_problem = 0
        for layout_index, layout_grasps in enumerate(candidate_grasps):
            if layout_grasps is None:
                continue
            problems = slice(next_problem, next_problem + len(layout_grasps.targets))
            next_problem = problems.stop
            ik_per_layout[layout_index] = IKFeasibility(
                feasible=feasible[problems],
                position_error=position_error[problems],
                rotation_error=rotation_error[problems],
                joint_positions=joint_positions[problems],
            )
        return ik_per_layout

    def _visualize_layout(
        self, layout_grasps: CandidateLayoutGrasps, ik: IKFeasibility, layout_index_within_batch: int
    ) -> None:
        """Draw one layout's grasps and IK errors in the placement visualizer, when it is on."""
        if self._rerun_layer is None:
            return
        layout_index_across_batch = self._visualizer.get_layout_index_across_batch(layout_index_within_batch)
        self._rerun_layer.log_layout(
            layout_index_across_batch=layout_index_across_batch,
            robot_base_pos_w=layout_grasps.robot_base_pose_w.position_xyz,
            robot_base_quat_w_xyzw=layout_grasps.robot_base_pose_w.rotation_xyzw,
            target_names=[obj.name for obj in layout_grasps.targets],
            grasp_poses_base_frame=layout_grasps.grasp_poses,
            feasible=ik.feasible,
            position_error=ik.position_error,
            rotation_error=ik.rotation_error,
            # The pose the arm ended up in is what explains a rejection, so it is drawn alongside it.
            robot_spheres=robot_collision_spheres(self._solver, ik.joint_positions),
            muted_sphere_mask=hand_sphere_mask(self._solver) if self._require_collision_free else None,
        )

    @staticmethod
    def _record_failure(rejections: list[Rejection], targets: list[ObjectBase], ik: IKFeasibility) -> None:
        """Record a rejection for each target whose top-down grasp has no feasible IK solution."""
        for target_index, target in enumerate(targets):
            if not ik.feasible[target_index].item():
                rejections.append(
                    Rejection(
                        f"'{target.name}' {PlacementCheck.IK_REACHABLE}: top-down grasp unreachable",
                        detail=(
                            f"IK position error {ik.position_error[target_index].item():.3f} m,"
                            f" rotation error {ik.rotation_error[target_index].item():.3f} rad"
                        ),
                    )
                )

    def _select_reachability_targets(self, objects: list[ObjectBase], anchors: set[ObjectBase]) -> list[ObjectBase]:
        """Movable objects the task marked as reachability targets (carry a RequiresReachability relation)."""
        return [obj for obj in objects if obj not in anchors and obj.has_relation(RequiresReachability)]
