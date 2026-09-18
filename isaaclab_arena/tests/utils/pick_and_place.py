# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script object lifts through public simulation APIs for pick-and-place tests."""

import torch


def lift_settled_objects_once(env, object_name: str, lifted_envs: torch.Tensor) -> None:
    """Lift newly settled objects once in each selected environment, then let physics place them."""
    from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state

    _, has_settled = get_object_initial_rest_state(env, object_name)
    ready_env_ids = (has_settled & ~lifted_envs).nonzero(as_tuple=False).flatten()
    if ready_env_ids.numel() == 0:
        return

    object_asset = env.scene[object_name]
    # W is the simulation world; O is the object frame.
    T_W_O = env.arena_world.get_pose_w(object_name)[ready_env_ids].clone()
    T_W_O[:, 2] += 0.12
    object_asset.write_root_pose_to_sim(T_W_O, env_ids=ready_env_ids)
    object_asset.write_root_velocity_to_sim(
        torch.zeros((ready_env_ids.numel(), 6), device=env.device),
        env_ids=ready_env_ids,
    )
    lifted_envs[ready_env_ids] = True
