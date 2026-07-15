# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Mapping
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_entity import PlacementEntity
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer

IDENTITY_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

# Name of the reset event term that owns the pooled object placer.
PLACEMENT_RESET_EVENT_NAME = "placement_reset"


def get_placement_pool(env) -> PooledObjectPlacer | None:
    """Return the pooled placer stored on the env reset event, or ``None`` when absent.

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


def get_placement_scene_entity_names(env) -> Mapping[str, str] | None:
    """Return the scene-name map stored on the placement reset event."""
    try:
        term_cfg = env.unwrapped.event_manager.get_term_cfg(PLACEMENT_RESET_EVENT_NAME)
    except ValueError:
        return None
    return term_cfg.params.get("scene_entity_names")


def get_rotation_xyzw(obj: PlacementEntity) -> tuple[float, float, float, float]:
    """Return the RotateAroundSolution rotation for *obj*, or identity if none."""
    rotate_marker = next((r for r in obj.get_relations() if isinstance(r, RotateAroundSolution)), None)
    return rotate_marker.get_rotation_xyzw() if rotate_marker else IDENTITY_ROTATION_XYZW


def get_base_rotation_per_object(
    objects: list[PlacementEntity],
) -> dict[PlacementEntity, tuple[float, float, float, float]]:
    """Return the base rotation for each object."""
    return {obj: get_rotation_xyzw(obj) for obj in objects}


def get_pose_from_layout(obj: PlacementEntity, layout: PlacementResult) -> Pose:
    """Return an entity pose from a solved layout."""
    assert obj in layout.positions, f"Placement layout is missing non-anchor entity '{obj.name}'"
    base_rotation = get_rotation_xyzw(obj)
    marker_yaw = yaw_from_quat_xyzw(base_rotation)
    total_yaw = layout.orientations.get(obj, marker_yaw)
    rotation = rotate_quat_by_yaw(base_rotation, total_yaw - marker_yaw)
    return Pose(position_xyz=layout.positions[obj], rotation_xyzw=rotation)


def get_movable_object_names(
    objects: list[PlacementEntity],
    anchor_objects_set: set[PlacementEntity],
    scene_entity_names: Mapping[str, str],
) -> list[str]:
    """Return scene names for non-anchor placement entities."""
    return [scene_entity_names[obj.name] for obj in objects if obj not in anchor_objects_set]


def write_layout_to_sim(
    env: ManagerBasedEnv,
    env_id: int,
    result: PlacementResult,
    anchor_objects_set: set[PlacementEntity],
    base_rotations: dict[PlacementEntity, tuple[float, float, float, float]],
    scene_entity_names: Mapping[str, str] | None = None,
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
        scene_entity_names: Isaac Lab scene name for each placement entity. Entity
            names are used directly when omitted.
    """
    if scene_entity_names is None:
        scene_entity_names = {obj.name: obj.name for obj in base_rotations}
    env_id_tensor = torch.tensor([env_id], device=env.device)
    zero_velocity = Velocity.zero().to_tensor(device=env.device).unsqueeze(0)
    missing_entities = [
        obj.name for obj in base_rotations if obj not in anchor_objects_set and obj not in result.positions
    ]
    assert not missing_entities, f"Placement layout is missing non-anchor entities: {missing_entities}"
    for obj in result.positions:
        if obj in anchor_objects_set:
            continue
        asset = env.scene[scene_entity_names[obj.name]]
        pose = get_pose_from_layout(obj, result)
        pose_tensor = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_tensor[0, :3] += env.scene.env_origins[env_id, :]
        asset.write_root_pose_to_sim(pose_tensor, env_ids=env_id_tensor)
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_id_tensor)


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    objects: list[PlacementEntity],
    placement_pool: PooledObjectPlacer,
    scene_entity_names: Mapping[str, str],
) -> None:
    """Coordinated reset event that draws layouts from the pool and writes poses.

    Registered as a single EventTermCfg(mode="reset"). Layouts are env-indexed:
    one layout is consumed for each requested absolute env id, so partial resets
    only advance the pools of the resetting envs.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        objects: Objects participating in relation solving.
        placement_pool: Runtime pool of solved placement layouts.
        scene_entity_names: Isaac Lab scene name for each placement entity.
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
                f"env {cur_env}; failed checks: {result.validation_results.get_failed_validation_check_names}."
            )
        # only write the non-anchor objects to the sim
        write_layout_to_sim(env, cur_env, result, anchor_objects_set, base_rotations, scene_entity_names)


def place_entities_from_layouts(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    objects: list[PlacementEntity],
    layouts: list[PlacementResult],
    scene_entity_names: Mapping[str, str],
) -> None:
    """Restore one fixed placement layout per environment.

    Args:
        env: The Isaac Lab environment.
        env_ids: Environment indices to restore.
        objects: Entities participating in relation solving.
        layouts: Fixed layout indexed by environment.
        scene_entity_names: Isaac Lab scene name for each placement entity.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    assert len(layouts) == env.scene.env_origins.shape[0], "Static layouts must match the scene environment count"
    anchor_objects_set = set(get_anchor_objects(objects))
    base_rotations = get_base_rotation_per_object(objects)
    for env_id in env_ids.tolist():
        write_layout_to_sim(
            env,
            env_id,
            layouts[env_id],
            anchor_objects_set,
            base_rotations,
            scene_entity_names,
        )
