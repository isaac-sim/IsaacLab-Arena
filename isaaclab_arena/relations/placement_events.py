# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import replace
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult
from isaaclab_arena.relations.relations import RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    objects: list[ObjectBase],
    placer_params: ObjectPlacerParams,
) -> None:
    """Coordinated reset event that re-runs the relation solver and writes fresh poses.

    Registered as a single ``EventTermCfg(mode="reset")`` that replaces per-object
    pose events for solver-managed objects. Each call solves for the subset of
    environments being reset, producing a new random layout per env.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer``. Seed is forced to
            ``None`` so each reset gets fresh randomness.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    num_reset_envs = len(env_ids)

    # Solver needs autograd, so temporarily exit any active inference_mode context.
    reset_solver_params = replace(placer_params.solver_params, save_position_history=False)
    reset_params = replace(
        placer_params,
        solver_params=reset_solver_params,
        apply_positions_to_objects=False,
        verbose=False,
        placement_seed=None,
    )
    placer = ObjectPlacer(params=reset_params)
    with torch.inference_mode(False):
        result = placer.place(objects, num_envs=num_reset_envs, result_per_env=True)

    if isinstance(result, MultiEnvPlacementResult):
        results_per_env = result.results
    else:
        results_per_env = [result]

    anchor_objects_set = set(get_anchor_objects(objects))
    rotations: dict[ObjectBase, tuple[float, float, float, float]] = {}
    for obj in objects:
        if obj not in anchor_objects_set:
            rotate_marker = next((r for r in obj.get_relations() if isinstance(r, RotateAroundSolution)), None)
            rotations[obj] = rotate_marker.get_rotation_xyzw() if rotate_marker else (0.0, 0.0, 0.0, 1.0)

    # Always write positions, even when validation failed — the solver's best-effort
    # is still better than the USD defaults (objects at origin).
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
