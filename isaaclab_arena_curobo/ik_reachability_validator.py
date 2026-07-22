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
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation import PlacementCheck, PlacementValidationResults
from isaaclab_arena.relations.placement_validators import PlacementValidator
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw
from isaaclab_arena_curobo.curobo_ik_utils import (
    SimFreeIKSolver,
    check_ik_feasibility,
    get_aabb_collision_cuboid_for_object,
    top_down_grasp_pose_from_world_poses,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams, ReachabilityConfig
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def get_object_world_pose_from_layout(
    result: PlacementResult,
    obj: ObjectBase,
    base_rotations: dict,
) -> tuple[tuple[float, float, float], tuple[float, ...]]:
    """Env-local world pose (position, xyzw quat) an object gets under a layout.

    Reconstructs the same root pose ``write_layout_to_sim`` would write: the layout position plus the
    object's marker rotation spun by its placement yaw.
    """
    pos_w = result.positions[obj]
    base_quat_xyzw = base_rotations[obj]
    marker_yaw = yaw_from_quat_xyzw(base_quat_xyzw)
    total_yaw = result.orientations.get(obj, marker_yaw)
    quat_w_xyzw = rotate_quat_by_yaw(base_quat_xyzw, total_yaw - marker_yaw)
    return pos_w, tuple(quat_w_xyzw)


def make_ik_reachability_validator(
    embodiment: EmbodimentBase,
    grasp_z_offset: float = 0.02,
    ik_pos_threshold: float = 0.01,
    ik_rot_threshold: float = 0.1,
    device: str | torch.device | None = None,
    stamp_results: bool = True,
) -> Callable[[PlacementResult], bool]:
    """Return a ``validator`` callable gating each layout on cuRobo top-down-grasp IK reachability.

    Args:
        embodiment: Embodiment who has a registered CuroboEmbodimentCfg.
        grasp_z_offset: Height (m) above each object's root for the top-down grasp pose.
        ik_pos_threshold: Max IK position error (m) for a grasp to count as reachable.
        ik_rot_threshold: Max IK rotation error (rad) for a grasp to count as reachable.
        device: Explicit CUDA device (e.g. ``"cuda:0"``); defaults to the current device.
        stamp_results: Record the verdict as a required ``PlacementCheck.IK_REACHABLE`` check on each
            layout, so it shows up in the audit report and gates that layout's ``.success``.
    """
    solver = SimFreeIKSolver(
        embodiment_curobo_cfg(embodiment),
        device=device,
        position_threshold=ik_pos_threshold,
        rotation_threshold=ik_rot_threshold,
    )
    base_pose = embodiment.get_base_pose()
    base_pos, base_quat_xyzw = base_pose.position_xyz, base_pose.rotation_xyzw

    def validator(result: PlacementResult) -> bool:
        objects = list(result.positions.keys())
        anchors = set(get_anchor_objects(objects))
        base_rotations = get_base_rotation_per_object(objects)

        world_poses = {obj: get_object_world_pose_from_layout(result, obj, base_rotations) for obj in objects}
        cuboids = [get_aabb_collision_cuboid_for_object(obj, *world_poses[obj]) for obj in objects]
        solver.update_world(cuboids, base_pos, base_quat_xyzw)

        movable = [obj for obj in objects if obj not in anchors]
        if not movable:
            reachable = True
        else:
            grasp_poses = torch.stack([
                top_down_grasp_pose_from_world_poses(
                    world_poses[obj][0],
                    world_poses[obj][1],
                    base_pos,
                    base_quat_xyzw,
                    grasp_z_offset,
                    device=solver.device,
                )
                for obj in movable
            ])
            feasible, _, _ = check_ik_feasibility(
                solver,
                grasp_poses,
                position_threshold=ik_pos_threshold,
                rotation_threshold=ik_rot_threshold,
            )
            reachable = bool(feasible.all().item())

        if stamp_results:
            result.validation_results.add_validation_check(PlacementCheck.IK_REACHABLE, reachable, required=True)
        return reachable

    return validator


def embodiment_curobo_cfg(embodiment: EmbodimentBase):
    """Look up the embodiment's registered cuRobo config (raises if none is registered)."""
    from isaaclab_arena_curobo.embodiment_curobo_registry import get_curobo_cfg_for

    return get_curobo_cfg_for(embodiment)


class ReachabilityValidator(PlacementValidator):
    """Build-time placement gate: the robot can reach a top-down grasp at every movable object (cuRobo IK).

    Wraps the reachability predicate as a gated ``PlacementValidator`` so the placer runs it only on
    candidates that already pass the cheaper geometry checks. Built via ``build_reachability_validator`` and
    injected through ``ObjectPlacerParams.extra_validators``; core placement never imports cuRobo itself.
    """

    check = PlacementCheck.IK_REACHABLE
    gated = True

    def __init__(self, predicate: Callable[[PlacementResult], bool]) -> None:
        self._predicate = predicate

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
        """Run the wrapped predicate over a single candidate's solved poses."""
        candidate = PlacementResult(
            validation_results=PlacementValidationResults(),
            positions=positions,
            final_loss=0.0,
            attempts=0,
            orientations=orientations,
        )
        return bool(self._predicate(candidate))


def build_reachability_validator(
    embodiment: EmbodimentBase, config: ReachabilityConfig | None = None
) -> ReachabilityValidator:
    """Build the cuRobo IK-reachability placement validator for an embodiment.

    Raises when cuRobo or the embodiment's registered cuRobo config is unavailable.

    Args:
        embodiment: Embodiment that has a registered CuroboEmbodimentCfg.
        config: Reachability tuning; defaults are used when omitted.
    """
    from isaaclab_arena.relations.object_placer_params import ReachabilityConfig

    config = config or ReachabilityConfig()
    predicate = make_ik_reachability_validator(
        embodiment,
        grasp_z_offset=config.grasp_z_offset_m,
        ik_pos_threshold=config.ik_position_threshold_m,
        ik_rot_threshold=config.ik_rotation_threshold_rad,
        stamp_results=False,
    )
    return ReachabilityValidator(predicate)


def configure_reachability_gate(placer_params: ObjectPlacerParams, embodiment: EmbodimentBase | None) -> bool:
    """Install the cuRobo IK-reachability gate on placer_params.extra_validators and report whether it could run.

    Args:
        placer_params: Placement params to receive the validator; its reachability_config tunes the gate.
        embodiment: Robot embodiment the grasps must be reachable by.
    """
    if embodiment is None:
        return False
    try:
        validator = build_reachability_validator(embodiment, placer_params.reachability_config)
    except AssertionError:
        # The embodiment has no registered cuRobo config -- treat reachability as unavailable.
        return False
    kept = [v for v in placer_params.extra_validators if v.check != PlacementCheck.IK_REACHABLE]
    # Reachability requires both embodiment curobo cfg and the solver deps to be importable.
    # If both exists, the validator implementation is added to placer_params.extra_validators.
    placer_params.extra_validators = [*kept, validator]
    return True
