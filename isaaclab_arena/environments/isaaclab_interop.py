# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena_environments.cli import ensure_environments_registered


def is_simulation_app_running() -> bool:
    """Checks if the simulation app is running."""
    import omni.kit.app

    try:
        app = omni.kit.app.get_app()
        return app is not None and app.is_running()
    except Exception:
        return False


def environment_registration_callback() -> list[str]:
    """This function is for use with Isaac Lab scripts to register an IsaacLab Arena environment.

    This function is passed to an Isaac Lab script as an external callback function. Example:

    python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py
        --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback
        --task lift_object
        --num_envs 512
        --object cracker_box
        agent.policy.activation=relu

    In this case the "lift_object" environment is registered with Isaac Lab before
    running the RSL RL training script. The training script will then run the
    training for the lift_object environment. In the example above we
    also use an environment flag to set the object to be a cracker box and
    Hydra to set the policy activation to be ReLU.

    """
    from isaaclab.app import AppLauncher

    # Start the simulation app if it is not running.
    if not is_simulation_app_running():
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args()
        AppLauncher(args)

    # Imports after the simulation app is started.
    from isaaclab_arena.cli.isaaclab_arena_cli import add_isaac_lab_cli_args, add_isaaclab_arena_cli_args
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # Get the requested environment from the CLI.
    parser = argparse.ArgumentParser()
    # NOTE(alexmillane, 2026.02.12): With the Isaac Lab interop, we use the task name to
    # determine the environment to register. The environment is also registered under this name.
    # The result is that a single argument tells Arena what to register, and Lab what to run.
    parser.add_argument("--task", type=str, required=True, help="Name of the IsaacLab Arena environment to register.")
    environment_name = parser.parse_known_args()[0].task
    ensure_environments_registered()
    environment = EnvironmentRegistry().get_component_by_name(environment_name)()
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
