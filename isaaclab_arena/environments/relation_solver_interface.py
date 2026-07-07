# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_rotation_xyzw, solve_and_place_objects
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv, rotate_quat_by_yaw, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.object_base import ObjectBase


def solve_and_apply_relation_placement(
    objects: list[ObjectBase],
    num_envs: int,
    placer_params: ObjectPlacerParams | None = None,
) -> EventTermCfg | None:
    """Solve relation placement and return a reset EventTermCfg (or None if no objects)."""
    objects = list(objects)
    if not objects:
        print("No objects with relations found in scene. Skipping relation solving.")
        return None

    if placer_params is None:
        placer_params = ObjectPlacerParams()
    else:
        placer_params = copy.copy(placer_params)
    placer_params.apply_positions_to_objects = False

    # TODO(xinjieyao, 2026-05-22): Add joint object/embodiment placement once task-dependent
    # reachability constraints are available. For now this always uses the object-only placer.
    placement_pool = PooledObjectPlacer(
        objects=objects,
        placer_params=placer_params,
        pool_size=num_envs * placer_params.min_unique_layouts_per_env,
        num_envs=num_envs,
    )

    if placement_pool.had_fallbacks:
        print(
            "Warning: Relation placement pool accepted best-loss fallback layouts "
            "that failed strict placement validation."
        )

    return _apply_relation_placement_result(
        objects=objects,
        placer_params=placer_params,
        placement_pool=placement_pool,
        num_envs=num_envs,
    )


def _apply_relation_placement_result(
    objects: list[ObjectBase],
    placer_params: ObjectPlacerParams,
    placement_pool: PooledObjectPlacer,
    num_envs: int,
) -> EventTermCfg | None:
    """Apply selected layouts to object spawn state and build reset event config."""
    anchor_objects_set = set(get_anchor_objects(objects))
    # Prevent external pose-reset events from conflicting with relation-solved objects.
    _validate_no_conflicting_pose_reset_events(objects, anchor_objects_set)

    # Anchor objects do not move, so no need to apply reset event.
    if anchor_objects_set == set(objects):
        return None

    if placer_params.resolve_on_reset:
        return _apply_dynamic_spawn_pose(
            objects=objects,
            placement_pool=placement_pool,
            anchor_objects_set=anchor_objects_set,
        )

    _apply_static_initial_poses(
        objects=objects,
        placement_pool=placement_pool,
        anchor_objects_set=anchor_objects_set,
        num_envs=num_envs,
    )
    return None


def _apply_dynamic_spawn_pose(
    objects: list[ObjectBase],
    placement_pool: PooledObjectPlacer,
    anchor_objects_set: set[ObjectBase],
) -> EventTermCfg:
    """Set initial spawn pose from one layout and return the reset placement event."""
    from isaaclab.managers import EventTermCfg

    # For env-indexed pools this seeds from env 0; the first reset overwrites with per-env layouts.
    layout = placement_pool.sample_with_replacement(1)[0]
    for obj in objects:
        if obj in anchor_objects_set:
            continue
        pos = layout.positions.get(obj)
        if pos is None:
            continue
        base_rot = get_rotation_xyzw(obj)
        marker_yaw = yaw_from_quat_xyzw(base_rot)
        total_yaw = layout.orientations.get(obj, marker_yaw)
        rot = rotate_quat_by_yaw(base_rot, total_yaw - marker_yaw)
        object_cfg = getattr(obj, "object_cfg", None)
        assert object_cfg is not None, f"Object '{obj.name}' must have object_cfg initialized before placement."
        object_cfg.init_state.pos = pos
        object_cfg.init_state.rot = rot

    return EventTermCfg(
        func=solve_and_place_objects,
        mode="reset",
        params={
            "objects": objects,
            "placement_pool": placement_pool,
        },
    )


def _apply_static_initial_poses(
    objects: list[ObjectBase],
    placement_pool: PooledObjectPlacer,
    anchor_objects_set: set[ObjectBase],
    num_envs: int,
) -> None:
    """Apply fixed per-environment poses for ``resolve_on_reset=False``."""
    layouts = placement_pool.sample_with_replacement(num_envs)
    for obj in objects:
        if obj in anchor_objects_set:
            continue
        base_rotation_xyzw = get_rotation_xyzw(obj)
        poses = []
        missing_envs: list[int] = []
        for env_idx in range(num_envs):
            pos = layouts[env_idx].positions.get(obj)
            if pos is None:
                missing_envs.append(env_idx)
            else:
                marker_yaw = yaw_from_quat_xyzw(base_rotation_xyzw)
                total_yaw = layouts[env_idx].orientations.get(obj, marker_yaw)
                rotation_xyzw = rotate_quat_by_yaw(base_rotation_xyzw, total_yaw - marker_yaw)
                poses.append(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
        if missing_envs:
            print(
                f"Warning: Object '{obj.name}' is missing positions in {len(missing_envs)} env(s) "
                f"(env ids: {missing_envs}); skipping set_initial_pose for this object."
            )
        else:
            obj.set_initial_pose(PosePerEnv(poses=poses))


def _validate_no_conflicting_pose_reset_events(
    objects: list[ObjectBase],
    anchor_objects_set: set[ObjectBase],
) -> None:
    """Reject conflicting explicit pose-reset events on relation-solved objects."""
    for obj in objects:
        assert not (obj not in anchor_objects_set and getattr(obj, "event_cfg", None) is not None), (
            f"Non-anchor object '{obj.name}' has an explicit pose-reset event. "
            "Relational solving should not be combined with explicit setting of "
            "poses on non-anchor objects."
        )
