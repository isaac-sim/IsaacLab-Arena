# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.cli import ExampleEnvironments


def environment_registration_callback() -> list[str]:
    """This function is for use with Isaac Lab scripts to register an IsaacLab Arena environment.

    This function is passed to an Isaac Lab script as an external callback function. Example:

    python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py
        --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback
        --task lift_object
        --num_envs 512

    In this case the "lift_object" environment is registered with Isaac Lab before
    running the RSL RL training script. The training script will then run the
    training for the lift_object environment.

    """
    from isaaclab.app import AppLauncher

    from isaaclab_arena.cli.isaaclab_arena_cli import add_isaac_lab_cli_args, add_isaaclab_arena_cli_args
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # Get the requested environment from the CLI.
    parser = argparse.ArgumentParser()
    # NOTE(alexmillane, 2026.02.12): With the Isaac Lab interop, we use the task name to
    # determine the environment to register. The environment is also registered under this name.
    # The result is that a single argument tells Arena what to register, and Lab what to run.
    parser.add_argument("--task", type=str, required=True, help="Name of the IsaacLab Arena environment to register.")
    environment_name = parser.parse_known_args()[0].task
    environment = ExampleEnvironments[environment_name]()
    # Get the full list of environment-specific CLI args.
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)
    environment.add_cli_args(parser)
    args, remaining_args = parser.parse_known_args()
    # Create the environment config
    isaaclab_arena_environment = environment.get_env(args)
    # Build and register the environment
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args)
    env_builder.build_registered()
    # Return the arguments that were not consumed by this callback
    return remaining_args
