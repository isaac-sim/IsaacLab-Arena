# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

# from isaaclab_arena_environments.lift_object_environment import LiftObjectEnvironment
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena_environments.cli import ExampleEnvironments


# def add_environment_registration_args(parser: argparse.ArgumentParser) -> argparse.ArgumentGroup:
# def get_environment(argv: list[str]) -> tuple[IsaacLabArenaEnvironment, list[str]]:
def get_environment(
    arena_environment_name: str, remaining_args: list[str]
) -> tuple[IsaacLabArenaEnvironment, list[str]]:
    # Get the environment class
    environment = ExampleEnvironments[arena_environment_name]()
    # Get arguments associated with this environment
    parser = argparse.ArgumentParser()
    environment.add_cli_args(parser)
    # args, remaining_args = parser.parse_known_args(remaining_args)
    args, remaining_args = parser.parse_known_args(remaining_args)
    # Build the environment (from the args)
    isaaclab_arena_environment = environment.get_env(args)
    return isaaclab_arena_environment, remaining_args


# def my_env_registration_callback(argv: list[str]) -> list[str]:
def my_env_registration_callback() -> list[str]:
    """Parse arena-specific CLI args, register env, and return remaining args.

    This function is designed to be called from the main training script with
    the list of arguments that have not yet been consumed. It parses only the
    arguments it knows about and returns the leftover list for downstream use
    (e.g. Hydra / other parsers).
    """

    from isaaclab.app import AppLauncher

    from isaaclab_arena.cli.isaaclab_arena_cli import add_isaac_lab_cli_args, add_isaaclab_arena_cli_args
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # print("Hello from my_env_registration_callback!")
    # Build parser for arena-specific CLI args and parse only from the provided argv
    # parser = get_isaaclab_arena_cli_parser()
    # LiftObjectEnvironment.add_cli_args(parser)
    # args, remaining_args = parser.parse_known_args(argv)
    # # args, _ = get_isaaclab_arena_environments_cli_parser().parse_known_args()
    # print(f"args: {args}")
    # isaaclab_arena_environment = LiftObjectEnvironment().get_env(args)
    # print(f"isaaclab_arena_environment: {isaaclab_arena_environment}")
    # isaaclab_arena_environment, remaining_args = get_environment(argv)
    parser = argparse.ArgumentParser()
    # NOTE(alexmillane, 2026.02.12): With the Isaac Lab interop, we use the task name to
    # determine the environment to register. The environment is also registered under this name.
    # The result is that a single arugment tells Arena what to register, and Lab what to run.
    parser.add_argument("--task", type=str, required=True, help="Name of the IsaacLab Arena environment to register.")
    # Get the environment class
    environment_name = parser.parse_known_args()[0].task
    environment = ExampleEnvironments[environment_name]()
    # Get the full list of arguments
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)
    environment.add_cli_args(parser)
    args, remaining_args = parser.parse_known_args()

    # Get the environment
    # isaaclab_arena_environment, remaining_args = get_environment(args.task, remaining_args)
    isaaclab_arena_environment = environment.get_env(args)

    # Build and register the environment (from the args)
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args)
    env_builder.build_registered()
    # print(f"env_cfg: {env_cfg}")
    # Return only the arguments that were not consumed by this callback
    return remaining_args
