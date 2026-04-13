# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
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
    if env_ids is None:
        return

    num_reset_envs = len(env_ids)

    # Re-solve with seed=None (fresh randomness) and apply_positions_to_objects=False
    # because we write poses directly to the simulation below.
    # Exit inference_mode because policy_runner wraps env.step() in
    # torch.inference_mode(), which disables autograd — but the solver needs it.
    reset_params = ObjectPlacerParams(
        solver_params=placer_params.solver_params,
        max_placement_attempts=placer_params.max_placement_attempts,
        apply_positions_to_objects=False,
        verbose=False,
        placement_seed=None,
        min_separation_m=placer_params.min_separation_m,
        on_relation_z_tolerance_m=placer_params.on_relation_z_tolerance_m,
    )
    placer = ObjectPlacer(params=reset_params)
    with torch.inference_mode(False):
        result = placer.place(objects, num_envs=num_reset_envs, result_per_env=True)

    if isinstance(result, MultiEnvPlacementResult):
        results_per_env = result.results
    else:
        results_per_env = [result]

    # Identify anchors (fixed references) — their poses are not written.
    anchor_objects_set = set(get_anchor_objects(objects))

    # Pre-compute rotation for each non-anchor object (same across all envs).
    # Pattern matches ObjectPlacer._apply_positions.
    rotations: dict[str, tuple[float, float, float, float]] = {}
    for obj in objects:
        if obj not in anchor_objects_set:
            rotate_marker = next((r for r in obj.get_relations() if isinstance(r, RotateAroundSolution)), None)
            rotations[obj.name] = rotate_marker.get_rotation_xyzw() if rotate_marker else (0.0, 0.0, 0.0, 1.0)

    # Write solved poses into each resetting environment.
    # Pattern follows set_object_pose_per_env in isaaclab_arena/terms/events.py.
    for local_idx, cur_env in enumerate(env_ids.tolist()):
        positions = results_per_env[local_idx].positions
        for obj, pos in positions.items():
            if obj in anchor_objects_set:
                continue
            asset = env.scene[obj.name]
            pose = Pose(position_xyz=pos, rotation_xyzw=rotations[obj.name])
            pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
            pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[cur_env, :]
            asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=torch.tensor([cur_env], device=env.device))
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )
