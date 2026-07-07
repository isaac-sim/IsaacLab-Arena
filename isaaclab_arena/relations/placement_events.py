# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv

from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.pose import Pose, rotate_quat_by_yaw, yaw_from_quat_xyzw
from isaaclab_arena.utils.velocity import Velocity

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.placement_result import PlacementResult

IDENTITY_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

# Name of the reset event term that owns the pooled object placer
PLACEMENT_RESET_EVENT_NAME = "placement_reset"


def get_placement_pool(env) -> PooledObjectPlacer | None:
    """Return the pooled placer registered on the env, or ``None`` when the env has no pooled placement.

    Lets a runtime caller reach the pool (e.g. to run the post-reset settle check) from the env alone,
    without holding the builder. The pool is reached through the env's event manager.

    Args:
        env: The gym-wrapped Isaac Lab env; the base env is reached via ``env.unwrapped``.
    """
    try:
        term_cfg = env.unwrapped.event_manager.get_term_cfg(PLACEMENT_RESET_EVENT_NAME)
    except ValueError:
        return None
    return term_cfg.params.get("placement_pool")


def get_rotation_xyzw(obj: ObjectBase) -> tuple[float, float, float, float]:
    """Return the RotateAroundSolution rotation for *obj*, or identity if none."""
    rotate_marker = next((r for r in obj.get_relations() if isinstance(r, RotateAroundSolution)), None)
    return rotate_marker.get_rotation_xyzw() if rotate_marker else IDENTITY_ROTATION_XYZW


def get_base_rotation_per_object(objects: list[ObjectBase]) -> dict[ObjectBase, tuple[float, float, float, float]]:
    """Return the base rotation for each object."""
    return {obj: get_rotation_xyzw(obj) for obj in objects}


def get_movable_object_names(
    objects: list[ObjectBase],
    anchor_objects_set: set[ObjectBase],
) -> list[str]:
    """Return the names of non-anchor objects."""
    return [obj.name for obj in objects if obj not in anchor_objects_set]


def write_layout_to_sim(
    env: ManagerBasedEnv,
    env_id: int,
    result: PlacementResult,
    anchor_objects_set: set[ObjectBase],
    base_rotations: dict[ObjectBase, tuple[float, float, float, float]],
) -> None:
    """Write one env's solved layout into the sim.

    Even writing zero velocity, the sim will still apply gravity and other forces from collisions,
    so collided objects will still be subject to move.

    Args:
        env: The Isaac Lab ManagerBasedEnv environment.
        env_id: The environment index.
        result: The placement result to write to the sim.
        anchor_objects_set: The set of anchor objects.
        base_rotations: The base rotations for all objects.
    """
    env_id_tensor = torch.tensor([env_id], device=env.device)
    zero_velocity = Velocity.zero().to_tensor(device=env.device).unsqueeze(0)
    for obj, pos in result.positions.items():
        if obj in anchor_objects_set:
            continue
        asset = env.scene[obj.name]
        marker_yaw = yaw_from_quat_xyzw(base_rotations[obj])
        total_yaw = result.orientations.get(obj, marker_yaw)
        rotation_xyzw = rotate_quat_by_yaw(base_rotations[obj], total_yaw - marker_yaw)
        pose = Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw)
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[env_id, :]
        asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_id_tensor)
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_id_tensor)


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    objects: list[ObjectBase],
    placement_pool: PooledObjectPlacer,
) -> None:
    """Coordinated reset event that draws layouts from the pool and writes poses.

    Registered as a single EventTermCfg(mode="reset"). Layouts are env-indexed:
    one layout is consumed for each requested absolute env id, so partial resets
    only advance the pools of the resetting envs.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        objects: All objects (including anchors) participating in relation solving.
        placement_pool: Pooled object placer to draw layouts from.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    reset_env_ids = env_ids.tolist()
    num_scene_envs = env.scene.env_origins.shape[0]
    assert (
        placement_pool.num_envs == num_scene_envs
    ), f"Placement pool has {placement_pool.num_envs} envs, but scene has {num_scene_envs} env origins."
    results_by_env = placement_pool.sample_for_envs(reset_env_ids)

    anchor_objects_set = set(get_anchor_objects(objects))
    base_rotations = get_base_rotation_per_object(objects)

    for cur_env in reset_env_ids:
        result = results_by_env[cur_env]
        if not result.success:
            print(
                "Warning: Writing best-loss fallback placement for "
                f"env {cur_env}; layout failed strict placement validation."
            )
        # only write the non-anchor objects to the sim
        write_layout_to_sim(env, cur_env, result, anchor_objects_set, base_rotations)
