# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from itertools import count
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv

from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.placement_result import PlacementResult

IDENTITY_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

# Name of the reset event term that owns the pooled object placer.
PLACEMENT_RESET_EVENT_NAME = "placement_reset"
# EventTermCfg params must stay config-serializable, so runtime pools live here
# and params carry only a registry key.
_PLACEMENT_POOL_REGISTRY: dict[str, PooledObjectPlacer] = {}
_PLACEMENT_POOL_KEY_COUNTER = count()


def register_placement_pool(placement_pool: PooledObjectPlacer) -> str:
    """Register a runtime placement pool and return its config-safe key."""
    key = f"placement_pool_{next(_PLACEMENT_POOL_KEY_COUNTER)}"
    _PLACEMENT_POOL_REGISTRY[key] = placement_pool
    return key


def unregister_placement_pool(key: str) -> None:
    """Remove a runtime placement pool registration when its env is torn down."""
    _PLACEMENT_POOL_REGISTRY.pop(key, None)


def clear_placement_pool_registry() -> None:
    """Clear all runtime placement pools, primarily for notebooks and tests."""
    _PLACEMENT_POOL_REGISTRY.clear()


def _get_registered_placement_pool(key: str) -> PooledObjectPlacer:
    """Return a registered placement pool or raise if the key is unavailable."""
    try:
        return _PLACEMENT_POOL_REGISTRY[key]
    except KeyError as exc:
        raise RuntimeError(
            f"Placement pool '{key}' is not registered in this process. "
            "Pooled placement reset events must be created and used in the same Python process."
        ) from exc


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
    placement_pool_key = term_cfg.params.get("placement_pool_key")
    if placement_pool_key is None:
        return None
    return _get_registered_placement_pool(placement_pool_key)


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
    _write_layout_to_sim_by_name(
        env,
        env_id,
        result,
        anchor_object_names={obj.name for obj in anchor_objects_set},
        rotations_by_name={obj.name: rotation for obj, rotation in base_rotations.items()},
    )


def _write_layout_to_sim_by_name(
    env: ManagerBasedEnv,
    env_id: int,
    result: PlacementResult,
    anchor_object_names: set[str],
    rotations_by_name: dict[str, tuple[float, float, float, float]],
) -> None:
    """Write one env's solved layout using only config-safe object names."""
    env_id_tensor = torch.tensor([env_id], device=env.device)
    zero_velocity = Velocity.zero().to_tensor(device=env.device).unsqueeze(0)
    for obj, pos in result.positions.items():
        if obj.name in anchor_object_names:
            continue
        asset = env.scene[obj.name]
        base_rotation = rotations_by_name[obj.name]
        marker_yaw = yaw_from_quat_xyzw(base_rotation)
        total_yaw = result.orientations.get(obj, marker_yaw)
        rotation_xyzw = rotate_quat_by_yaw(base_rotation, total_yaw - marker_yaw)
        pose = Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw)
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[env_id, :]
        asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_id_tensor)
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_id_tensor)


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_names: list[str],
    anchor_object_names: list[str],
    rotations_by_name: dict[str, tuple[float, float, float, float]],
    placement_pool_key: str,
) -> None:
    """Coordinated reset event that draws layouts from the pool and writes poses.

    Registered as a single EventTermCfg(mode="reset"). Layouts are env-indexed:
    one layout is consumed for each requested absolute env id, so partial resets
    only advance the pools of the resetting envs.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        object_names: Config-safe names of objects participating in relation solving.
        anchor_object_names: Config-safe names of anchor objects.
        rotations_by_name: Config-safe rotations keyed by object name.
        placement_pool_key: Key returned by register_placement_pool.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    placement_pool = _get_registered_placement_pool(placement_pool_key)
    missing_rotations = set(object_names) - set(rotations_by_name)
    assert not missing_rotations, f"Missing rotations for objects: {sorted(missing_rotations)}"

    reset_env_ids = env_ids.tolist()
    num_scene_envs = env.scene.env_origins.shape[0]
    assert (
        placement_pool.num_envs == num_scene_envs
    ), f"Placement pool has {placement_pool.num_envs} envs, but scene has {num_scene_envs} env origins."
    results_by_env = placement_pool.sample_for_envs(reset_env_ids)

    anchor_object_names_set = set(anchor_object_names)

    for cur_env in reset_env_ids:
        result = results_by_env[cur_env]
        if not result.success:
            print(
                "Warning: Writing best-loss fallback placement for "
                f"env {cur_env}; failed checks: {result.validation_results.get_failed_validation_check_names}."
            )
        # only write the non-anchor objects to the sim
        _write_layout_to_sim_by_name(env, cur_env, result, anchor_object_names_set, rotations_by_name)
