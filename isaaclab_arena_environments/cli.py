# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
from typing import TYPE_CHECKING

from isaaclab_arena.assets.asset_registry import EnvironmentRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder


def ensure_environments_registered():
    """Trigger registration of all environments in the ``isaaclab_arena_environments`` package.

    Importing the package fires the ``@register_environment`` decorator on each
    environment module, which handles registration.  The import is cached by
    Python, so subsequent calls are free.
    """
    import isaaclab_arena_environments  # noqa: F401


def parse_and_return_external_environment_from_string(
    environment_path: str,
) -> tuple[str, type[ExampleEnvironmentBase]]:
    """Parse a string and import the environment class

    Args:
        environment_path: The path to the environment class

    Raises:
        ValueError: If the environment path is not in the format "module_path:class_name"

    Returns:
        tuple[str, type[ExampleEnvironmentBase]]: A tuple with the environment name and the environment class
    """
    # Parse the environment path and import the environment class
    # We assume the environment path is in the format "module_path:class_name"
    # Add a check for the format
    if ":" not in environment_path:
        raise ValueError(f"Invalid environment path: {environment_path}. Expected format: 'module_path:class_name'")
    module_path, class_name = environment_path.split(":", 1)
    try:
        module = importlib.import_module(module_path)
        environment_class = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"Could not resolve the environment path '{environment_path}' into an environment class."
            " The format should be 'module_path:class_name'.\n"
            f"Received the error:\n {e}."
        ) from e
    name = getattr(environment_class, "name", environment_class.__name__)
    assert name is not None, "Environment class must have a 'name' attribute"
    return name, environment_class


def add_example_environments_cli_args(args_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()

    args, unknown = args_parser.parse_known_args()
    environment = getattr(args, "external_environment_class_path", None)
    if environment is not None:
        print(f"Adding external environment: {environment}")
        name, cls = parse_and_return_external_environment_from_string(environment)
        env_registry.register(cls, name)

    subparsers = args_parser.add_subparsers(
        dest="example_environment", required=True, help="Example environment to run"
    )
    for env_name in env_registry.get_all_keys():
        env_cls = env_registry.get_component_by_name(env_name)
        subparser = subparsers.add_parser(env_cls.name)
        env_cls.add_cli_args(subparser)

    return args_parser


def get_isaaclab_arena_environments_cli_parser(
    args_parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if args_parser is None:
        args_parser = get_isaaclab_arena_cli_parser()
    # NOTE(alexmillane, 2025.09.04): This command adds subparsers for each example environment.
    # So it has to be added last, because the subparser flags are parsed after the others.
    args_parser = add_example_environments_cli_args(args_parser)
    return args_parser


def get_arena_builder_from_cli(args_cli: argparse.Namespace) -> ArenaEnvBuilder:

    ensure_environments_registered()
    env_registry = EnvironmentRegistry()

    assert hasattr(args_cli, "example_environment"), "Example environment must be specified"
    assert env_registry.is_registered(
        args_cli.example_environment
    ), f"Example environment type {args_cli.example_environment} not supported"
    example_env = env_registry.get_component_by_name(args_cli.example_environment)()

    env_builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
    return env_builder
