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

import isaaclab.utils.math as math_utils

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
from isaaclab_arena_curobo.utils.frame_utils import top_down_grasp_matrices, world_poses_to_robot_frame_batched
from isaaclab_arena_curobo.utils.ik_solver_utils import (
    IKFeasibility,
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
        self._ik_batch_size = config.ik_batch_size
        assert self._ik_batch_size >= 1, f"ik_batch_size must be at least 1, got {self._ik_batch_size}"
        self._solver = CuroboIKSolver(
            get_embodiment_curobo_cfg(config.embodiment),
            position_threshold=self._ik_pos_threshold,
            rotation_threshold=self._ik_rot_threshold,
            max_batch_size=self._ik_batch_size,
            multi_env=True,
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
        """Whether the robot can reach a top-down grasp at the target objects of every candidate layout.

        Every (layout, target) grasp is one IK problem; the problems of all layouts are solved in batches of
        ``ik_batch_size``, each against a collision world holding the other objects of its own layout. A layout
        with nothing to grasp (anchor-only, or no target present) is trivially reachable. The robot stands where
        a layout puts it when it is relation-placed too, so the grasps are solved from the base pose the layout
        would actually spawn.
        """
        candidates = batch.candidates
        if not candidates:
            return []
        objects = list(candidates[0].positions.keys())
        anchors = set(get_anchor_objects(objects))
        targets = self._select_reachability_targets(objects, anchors)
        if not targets:
            if not self._warned_no_targets:
                print(
                    "[ReachabilityValidator] WARNING: enabled but resolved zero reachability targets; every layout "
                    "passes the IK check trivially. No reachability targets found in the task."
                )
                self._warned_no_targets = True
            return [True] * len(candidates)
        obstacles = [obj for obj in objects if obj is not self._embodiment]
        device = self._solver.device
        num_layouts, num_targets = len(candidates), len(targets)

        t_W_all, q_W_all, t_W_R, q_W_R = self._world_poses(candidates, objects)
        obstacle_index = {obj: objects.index(obj) for obj in obstacles}
        target_index = [objects.index(obj) for obj in targets]
        obstacle_columns = [obstacle_index[obj] for obj in obstacles]

        # Collision cuboids of every layout, in that layout's robot base frame: (L, K, ...).
        bbox_center_O = torch.stack([obj.get_bounding_box().center[0] for obj in obstacles]).to(device, torch.float32)
        bbox_dims = torch.stack([obj.get_bounding_box().size[0] for obj in obstacles]).to(device, torch.float32)
        q_W_O = q_W_all[:, obstacle_columns]
        R_W_O = math_utils.matrix_from_quat(q_W_O.reshape(-1, 4)).reshape(num_layouts, len(obstacles), 3, 3)
        center_W = t_W_all[:, obstacle_columns] + (R_W_O @ bbox_center_O.unsqueeze(-1)).squeeze(-1)
        t_R_O, q_R_O = world_poses_to_robot_frame_batched(
            center_W.reshape(-1, 3),
            q_W_O.reshape(-1, 4),
            t_W_R.repeat_interleave(len(obstacles), 0),
            q_W_R.repeat_interleave(len(obstacles), 0),
        )
        t_R_O = t_R_O.reshape(num_layouts, len(obstacles), 3)
        q_R_O = q_R_O.reshape(num_layouts, len(obstacles), 4)

        # Grasp poses of every (layout, target), in the robot base frame: (L * T, 4, 4), layout-major.
        t_R_T, q_R_T = world_poses_to_robot_frame_batched(
            t_W_all[:, target_index].reshape(-1, 3),
            q_W_all[:, target_index].reshape(-1, 4),
            t_W_R.repeat_interleave(num_targets, 0),
            q_W_R.repeat_interleave(num_targets, 0),
        )
        grasp_poses = top_down_grasp_matrices(t_R_T, q_R_T, self._grasp_z_offset)

        # One world per problem: the layout's obstacles minus the problem's own target.
        num_problems = num_layouts * num_targets
        problem_layout = torch.arange(num_layouts, device=device).repeat_interleave(num_targets)
        enabled = torch.ones(num_problems, len(obstacles), dtype=torch.bool, device=device)
        for target_position, target in enumerate(targets):
            if target in obstacle_index:
                enabled[target_position::num_targets, obstacles.index(target)] = False

        ik = self._solve_problems(grasp_poses, problem_layout, bbox_dims, t_R_O, q_R_O, enabled)

        feasible = ik.feasible.view(num_layouts, num_targets)
        verdicts = feasible.all(dim=1).tolist()
        for layout_index, candidate in enumerate(candidates):
            layout_slice = slice(layout_index * num_targets, (layout_index + 1) * num_targets)
            layout_ik = IKFeasibility(
                feasible=ik.feasible[layout_slice],
                position_error=ik.position_error[layout_slice],
                rotation_error=ik.rotation_error[layout_slice],
                joint_positions=ik.joint_positions[layout_slice],
            )
            if self._rerun_layer is not None:
                layout_index_across_batch = self._visualizer.get_layout_index_across_batch(layout_index)
                self._rerun_layer.log_layout(
                    layout_index_across_batch=layout_index_across_batch,
                    robot_base_pos_w=tuple(t_W_R[layout_index].tolist()),
                    robot_base_quat_w_xyzw=tuple(q_W_R[layout_index].tolist()),
                    target_names=[obj.name for obj in targets],
                    grasp_poses_base_frame=grasp_poses[layout_slice],
                    feasible=layout_ik.feasible,
                    position_error=layout_ik.position_error,
                    rotation_error=layout_ik.rotation_error,
                    robot_spheres=robot_collision_spheres(self._solver, layout_ik.joint_positions),
                    muted_sphere_mask=hand_sphere_mask(self._solver) if self._require_collision_free else None,
                )
            if not verdicts[layout_index]:
                self._record_failure(rejections, targets, layout_ik)
        return verdicts

    def _world_poses(
        self, candidates: list, objects: list[ObjectBase]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World poses of every object under every layout, and the robot base pose per layout.

        Returns ``(t_W_O (L, N, 3), q_W_O_xyzw (L, N, 4), t_W_R (L, 3), q_W_R_xyzw (L, 4))`` on the solver device.
        """
        base_rotations = get_base_rotation_per_asset(objects)
        positions, rotations = [], []
        robot_positions, robot_rotations = [], []
        for candidate in candidates:
            layout_positions, layout_rotations = [], []
            for obj in objects:
                pose = get_object_world_pose_from_layout(
                    candidate.positions, candidate.orientations, obj, base_rotations
                )
                layout_positions.append(pose.position_xyz)
                layout_rotations.append(pose.rotation_xyzw)
            positions.append(layout_positions)
            rotations.append(layout_rotations)
            robot_pose = self._configured_robot_base_pose_w
            if self._embodiment in candidate.positions:
                robot_pose = Pose(
                    position_xyz=layout_positions[objects.index(self._embodiment)],
                    rotation_xyzw=layout_rotations[objects.index(self._embodiment)],
                )
            robot_positions.append(robot_pose.position_xyz)
            robot_rotations.append(robot_pose.rotation_xyzw)
        device = self._solver.device
        return (
            torch.tensor(positions, dtype=torch.float32, device=device),
            torch.tensor(rotations, dtype=torch.float32, device=device),
            torch.tensor(robot_positions, dtype=torch.float32, device=device),
            torch.tensor(robot_rotations, dtype=torch.float32, device=device),
        )

    def _solve_problems(
        self,
        grasp_poses: torch.Tensor,
        problem_layout: torch.Tensor,
        bbox_dims: torch.Tensor,
        t_R_O: torch.Tensor,
        q_R_O: torch.Tensor,
        enabled: torch.Tensor,
    ) -> IKFeasibility:
        """IK-solve all grasp problems in batches of ``ik_batch_size``, each against its own layout's world.

        Every batch is padded to the full batch size by repeating its last problem, so cuRobo always sees the
        batch shape it was built for.
        """
        num_problems = grasp_poses.shape[0]
        batch_size = self._ik_batch_size
        feasible, position_error, rotation_error, joint_positions = [], [], [], []
        for start in range(0, num_problems, batch_size):
            problems = torch.arange(start, min(start + batch_size, num_problems), device=grasp_poses.device)
            padded = torch.cat([problems, problems[-1:].expand(batch_size - problems.numel())])
            layouts = problem_layout[padded]
            self._solver.load_cuboid_worlds(
                bbox_dims.unsqueeze(0).expand(batch_size, -1, -1), t_R_O[layouts], q_R_O[layouts], enabled[padded]
            )
            ik = solve_ik_feasibility(
                self._solver,
                grasp_poses[padded],
                position_threshold=self._ik_pos_threshold,
                rotation_threshold=self._ik_rot_threshold,
                require_collision_free=self._require_collision_free,
            )
            keep = problems.numel()
            feasible.append(ik.feasible[:keep])
            position_error.append(ik.position_error[:keep])
            rotation_error.append(ik.rotation_error[:keep])
            joint_positions.append(ik.joint_positions[:keep])
        return IKFeasibility(
            feasible=torch.cat(feasible),
            position_error=torch.cat(position_error),
            rotation_error=torch.cat(rotation_error),
            joint_positions=torch.cat(joint_positions),
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
