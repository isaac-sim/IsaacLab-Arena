# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv

from isaaclab_arena.relations.placement_pool import PlacementPool
from isaaclab_arena.relations.relations import RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    objects: list[ObjectBase],
    placement_pool: PlacementPool,
) -> None:
    """Coordinated reset event that draws layouts from a pre-solved pool and writes poses.

    Registered as a single ``EventTermCfg(mode="reset")``. Each call draws one
    layout per resetting environment from the pool and writes the poses to sim.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        objects: All objects (including anchors) participating in relation solving.
        placement_pool: Pre-solved pool of layouts to draw from.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    num_reset_envs = len(env_ids)
    results_per_env = placement_pool.draw(num_reset_envs)

    n_failed = sum(1 for r in results_per_env if not r.success)
    if n_failed > 0:
        print(f"[WARNING] Placement validation failed for {n_failed}/{num_reset_envs} envs. Writing best-effort positions.")

    anchor_objects_set = set(get_anchor_objects(objects))
    rotations: dict[ObjectBase, tuple[float, float, float, float]] = {}
    for obj in objects:
        if obj not in anchor_objects_set:
            rotate_marker = next((r for r in obj.get_relations() if isinstance(r, RotateAroundSolution)), None)
            rotations[obj] = rotate_marker.get_rotation_xyzw() if rotate_marker else (0.0, 0.0, 0.0, 1.0)

    zero_velocity = torch.zeros(1, 6, device=env.device)
    for local_idx, cur_env in enumerate(env_ids.tolist()):
        env_id_tensor = torch.tensor([cur_env], device=env.device)
        positions = results_per_env[local_idx].positions
        for obj, pos in positions.items():
            if obj in anchor_objects_set:
                continue
            asset = env.scene[obj.name]
            pose = Pose(position_xyz=pos, rotation_xyzw=rotations[obj])
            pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
            pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[cur_env, :]
            asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_id_tensor)
            asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_id_tensor)
