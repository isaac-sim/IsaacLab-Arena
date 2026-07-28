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

from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.placement_validator_registry import register_validator
from isaaclab_arena.relations.placement_validators import PlacementValidator
from isaaclab_arena.relations.relations import RequiresReachability, get_anchor_objects
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
from isaaclab_arena_curobo.ik_solver import CuroboIKSolver
from isaaclab_arena_curobo.utils.frame_utils import top_down_grasp_pose_from_world_poses
from isaaclab_arena_curobo.utils.ik_solver_utils import get_obb_collision_cuboid_for_object, solve_ik_feasibility

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.utils.bounding_box import OrientedBoundingBox


def get_object_world_pose_from_layout(
    positions: dict[ObjectBase, tuple[float, float, float]],
    rotations: dict[ObjectBase, tuple[float, float, float, float]],
    obj: ObjectBase,
    anchors: set[ObjectBase],
) -> Pose:
    """Return the world pose an object gets under a layout."""
    if obj in rotations:
        rotation_xyzw = rotations[obj]
    elif obj in anchors:
        fixed_pose = obj.get_initial_pose()
        assert isinstance(fixed_pose, Pose), f"Anchor '{obj.name}' must have a fixed Pose."
        rotation_xyzw = fixed_pose.rotation_xyzw
    else:
        rotation_xyzw = (0.0, 0.0, 0.0, 1.0)
    rotation = torch.tensor(rotation_xyzw, dtype=torch.float32)
    assert rotation.shape == (4,), f"Rotation for '{obj.name}' must have shape (4,)."
    assert torch.isfinite(rotation).all(), f"Rotation for '{obj.name}' must be finite."
    assert torch.isclose(
        torch.linalg.vector_norm(rotation), torch.tensor(1.0), atol=1e-5, rtol=1e-5
    ), f"Rotation for '{obj.name}' must be a unit quaternion."
    return Pose(
        position_xyz=tuple(float(v) for v in positions[obj]),
        rotation_xyzw=tuple(float(v) for v in rotation_xyzw),
    )


@register_validator
class ReachabilityValidator(PlacementValidator):
    """Build-time placement gate: the robot can reach a top-down grasp at the target objects (cuRobo IK).

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
        # Guards the zero-target warning so it fires once per validator, not once per candidate layout.
        self._warned_no_targets = False

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
        rotations: list[dict[ObjectBase, tuple[float, float, float, float]]],
        bboxes: list[dict[ObjectBase, OrientedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], rotations[i], bboxes[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[ObjectBase, tuple[float, float, float]],
        rotations: dict[ObjectBase, tuple[float, float, float, float]],
        bboxes: dict[ObjectBase, OrientedBoundingBox],
    ) -> bool:
        """Whether the robot can reach a top-down grasp at the target objects in one candidate layout.

        Rebuilds each object's world pose and a per-object collision cuboid, syncs them into the solver's
        world, then batches a single IK solve over the target objects' top-down grasps. A layout with
        nothing to grasp (anchor-only, or no target present) is trivially reachable.
        """
        objects = list(positions.keys())
        anchors = set(get_anchor_objects(objects))

        world_poses = {obj: get_object_world_pose_from_layout(positions, rotations, obj, anchors) for obj in objects}
        cuboids = [
            get_obb_collision_cuboid_for_object(
                obj,
                bboxes[obj],
                world_poses[obj].position_xyz,
                world_poses[obj].rotation_xyzw,
            )
            for obj in objects
        ]
        self._solver.update_world(cuboids, self._base_pos, self._base_quat_xyzw)

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
            for obj in targets
        ])
        feasible, _, _ = solve_ik_feasibility(
            self._solver,
            grasp_poses,
            position_threshold=self._ik_pos_threshold,
            rotation_threshold=self._ik_rot_threshold,
        )
        return bool(feasible.all().item())

    def _select_reachability_targets(self, objects: list[ObjectBase], anchors: set[ObjectBase]) -> list[ObjectBase]:
        """Movable objects the task marked as reachability targets (carry a RequiresReachability relation)."""
        return [obj for obj in objects if obj not in anchors and obj.has_relation(RequiresReachability)]
