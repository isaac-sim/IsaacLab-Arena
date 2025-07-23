# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import argparse
import gymnasium as gym

from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.environments.isaac_arena_manager_based_env import IsaacArenaManagerBasedRLEnvCfg

from isaaclab_tasks.utils import parse_env_cfg


def compile_arena_env_cfg(isaac_arena_environment: IsaacArenaEnvironment, args_cli: argparse.Namespace) -> gym.Env:
    # NOTE(cvolk): The scene apparently needs to hold a robot.
    # TODO(alex.millane, 2025-07-23): We're running into composition issues here.
    # move to using a more composable approach.
    scene_cfg = isaac_arena_environment.scene.get_scene_cfg()
    scene_cfg.robot = isaac_arena_environment.embodiment.get_robot_cfg()

    # Build the manager-based environment configuration.
    arena_env_cfg = IsaacArenaManagerBasedRLEnvCfg(
        observations=isaac_arena_environment.embodiment.get_observation_cfg(),
        actions=isaac_arena_environment.embodiment.get_action_cfg(),
        events=isaac_arena_environment.embodiment.get_event_cfg(),
        scene=scene_cfg,
        terminations=isaac_arena_environment.task.get_termination_cfg(),
    )

    gym.register(
        id=isaac_arena_environment.name,  # args_cli.task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": arena_env_cfg,
        },
        disable_env_checker=True,
    )
    env_cfg = parse_env_cfg(
        isaac_arena_environment.name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(isaac_arena_environment.name, cfg=env_cfg)

    # Reset for good measure.
    env.reset()

    return env
