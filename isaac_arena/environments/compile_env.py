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

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_tasks.utils import parse_env_cfg

from isaac_arena.assets.asset_registry import get_environment_configuration_from_registry
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.environments.isaac_arena_manager_based_env import IsaacArenaManagerBasedRLEnvCfg
from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaac_arena.utils.configclass import combine_configclass_instances


def get_arena_env_cfg(args_cli: argparse.Namespace) -> ManagerBasedRLEnvCfg:
    """Compile the arena environment configuration.

    Args:
        args_cli (argparse.Namespace): The command line arguments.

    Returns:
        ManagerBasedRLEnvCfg: The compiled arena environment configuration.
    """
    # Scene variation
    environment_configuration = get_environment_configuration_from_registry(
        args_cli.background, args_cli.object, args_cli.embodiment
    )

    # Arena Environment
    isaac_arena_environment = IsaacArenaEnvironment(
        name=f"pick_and_place_{args_cli.embodiment}_{args_cli.background}_{args_cli.object}",
        embodiment=environment_configuration["embodiment"],
        scene=PickAndPlaceScene(
            environment_configuration["background"],
            environment_configuration["object"],
        ),
        task=PickAndPlaceTask(),
    )

    # Compile an IsaacLab compatible arena environment configuration
    env_cfg = compile_environment_config(isaac_arena_environment, args_cli)

    return env_cfg, isaac_arena_environment.name


def compile_environment_config(
    isaac_arena_environment: IsaacArenaEnvironment, args_cli: argparse.Namespace
) -> ManagerBasedRLEnvCfg:
    """Compile the arena environment configuration to a gymnasium environment.

    Args:
        isaac_arena_environment (IsaacArenaEnvironment): The arena environment configuration.
        args_cli (argparse.Namespace): The command line arguments.

    Returns:
        gym.Env: The compiled gymnasium environment.
    """
    # Get the manager-based environment configuration.
    arena_env_cfg = compile_manager_based_env_cfg(isaac_arena_environment=isaac_arena_environment)
    if args_cli.mimic:
        # We compile the mimic env configuration now. This is a combination of the arena env cfg and the task mimic env cfg.
        task_mimic_env_cfg = isaac_arena_environment.task.get_mimic_env_cfg(
            embodiment_name=isaac_arena_environment.embodiment.name
        )
        env_cfg = combine_configclass_instances(
            "MimicEnvCfg",
            arena_env_cfg,
            task_mimic_env_cfg,
        )
        # We also point to the mimic env entry point which is a inherited class of ManagerBasedRLEnv.
        entry_point = isaac_arena_environment.embodiment.get_mimic_env()
    else:
        env_cfg = arena_env_cfg
        entry_point = "isaaclab.envs:ManagerBasedRLEnv"

    env_cfg = arena_to_gym_env_cfg(
        name=isaac_arena_environment.name, entry_point=entry_point, arena_env_cfg=env_cfg, args_cli=args_cli
    )

    return env_cfg


def compile_manager_based_env_cfg(isaac_arena_environment: IsaacArenaEnvironment) -> IsaacArenaManagerBasedRLEnvCfg:
    """Get the manager-based environment configuration.

    Args:
        isaac_arena_environment (IsaacArenaEnvironment): The arena environment configuration.

    Returns:
        IsaacArenaManagerBasedRLEnvCfg: The manager-based environment configuration.
    """

    # Set the robot position
    isaac_arena_environment.embodiment.set_robot_initial_pose(isaac_arena_environment.scene.get_robot_initial_pose())

    # Scene composition - The scene is composed of:
    # - Base IsaacLab config
    # - Contributions from the (background) scene
    # - Contributions from the embodiment
    scene_cfg = combine_configclass_instances(
        "SceneCfg",
        InteractiveSceneCfg(
            num_envs=4096,
            env_spacing=30.0,
            replicate_physics=False,
        ),
        isaac_arena_environment.scene.get_scene_cfg(),
        isaac_arena_environment.embodiment.get_scene_cfg(),
    )

    events_cfg = combine_configclass_instances(
        "EventsCfg",
        isaac_arena_environment.embodiment.get_event_cfg(),
        isaac_arena_environment.scene.get_events_cfg(),
    )

    termination_cfg = combine_configclass_instances(
        "TerminationCfg",
        isaac_arena_environment.task.get_termination_cfg(),
        isaac_arena_environment.scene.get_termination_cfg(),
    )

    # Build the manager-based environment configuration.
    arena_env_cfg = IsaacArenaManagerBasedRLEnvCfg(
        observations=isaac_arena_environment.embodiment.get_observation_cfg(),
        actions=isaac_arena_environment.embodiment.get_action_cfg(),
        events=events_cfg,
        scene=scene_cfg,
        terminations=termination_cfg,
        xr=isaac_arena_environment.embodiment.get_xr_cfg(),
    )
    return arena_env_cfg


def arena_to_gym_env_cfg(
    name: str,
    entry_point: str | ManagerBasedRLMimicEnv,
    arena_env_cfg: IsaacArenaManagerBasedRLEnvCfg,
    args_cli: argparse.Namespace,
) -> ManagerBasedRLEnvCfg:
    """Compile the arena environment configuration to a gymnasium environment.

    Args:
        name (str): The name of the environment.
        arena_env_cfg (IsaacArenaManagerBasedRLEnvCfg): The manager-based environment configuration.
        args_cli (argparse.Namespace): The command line arguments.

    Returns:
        gym.Env: The compiled gymnasium environment.
    """
    gym.register(
        id=name,
        entry_point=entry_point,
        kwargs={
            "env_cfg_entry_point": arena_env_cfg,
        },
        disable_env_checker=True,
    )
    env_cfg = parse_env_cfg(
        name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    return env_cfg


def make_gym_env(name: str, env_cfg: ManagerBasedRLEnvCfg) -> gym.Env:
    """Make the gymnasium environment.

    Args:
        name (str): The name of the environment.
        env_cfg (ManagerBasedRLEnvCfg): The environment configuration.

    Returns:
        gym.Env: The gymnasium environment.
    """

    env = gym.make(name, cfg=env_cfg)

    return env
