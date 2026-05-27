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
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase

IDENTITY_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)


def get_rotation_xyzw(obj: ObjectBase) -> tuple[float, float, float, float]:
    """Return the RotateAroundSolution rotation for *obj*, or identity if none."""
    rotate_marker = next((r for r in obj.get_relations() if isinstance(r, RotateAroundSolution)), None)
    return rotate_marker.get_rotation_xyzw() if rotate_marker else IDENTITY_ROTATION_XYZW


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    objects: list[ObjectBase],
    placement_pool: PooledObjectPlacer,
) -> None:
    """Coordinated reset event that draws layouts from the pooled placer and writes poses.

    Registered as a single ``EventTermCfg(mode="reset")``. Env-specific
    layouts advance by one full env round so each result still matches its
    absolute env id. Reusable layouts draw only for the environments being
    reset.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        objects: All objects (including anchors) participating in relation solving.
        placement_pool: Pooled placer to draw layouts from.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    reset_env_ids = env_ids.tolist()
    if placement_pool.requires_env_indexed_layouts:
        all_results = placement_pool.sample_without_replacement(env.scene.env_origins.shape[0])
        results_by_env = {cur_env: all_results[cur_env] for cur_env in reset_env_ids}
    else:
        reset_results = placement_pool.sample_without_replacement(len(reset_env_ids))
        results_by_env = dict(zip(reset_env_ids, reset_results))

    anchor_objects_set = set(get_anchor_objects(objects))
    rotations = {obj: get_rotation_xyzw(obj) for obj in objects if obj not in anchor_objects_set}

    zero_velocity = torch.zeros(1, 6, device=env.device)
    for cur_env in reset_env_ids:
        env_id_tensor = torch.tensor([cur_env], device=env.device)
        positions = results_by_env[cur_env].positions
        for obj, pos in positions.items():
            if obj in anchor_objects_set:
                continue
            asset = env.scene[obj.name]
            pose = Pose(position_xyz=pos, rotation_xyzw=rotations[obj])
            pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
            pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[cur_env, :]
            asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_id_tensor)
            asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_id_tensor)
