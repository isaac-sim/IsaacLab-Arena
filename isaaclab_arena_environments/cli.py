# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.cli.dataclass_cli import add_dataclass_cli_args, dataclass_from_cli
from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import CliOverrideSpec
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


def ensure_environments_registered():
    """Trigger registration of all environments in the ``isaaclab_arena_environments`` package.

    Importing the package fires the ``@register_environment`` decorator on each
    environment module, which handles registration.  The import is cached by
    Python, so subsequent calls are free.
    """
    import isaaclab_arena_environments  # noqa: F401


def import_registered_environment_factory(module_path: str, class_name: str) -> type[ArenaEnvironmentFactory]:
    """Import an environment factory defined outside this package and check its registration.

    Built-in environments are registered before the module is imported. A factory that
    reuses a built-in name or config type is therefore rejected instead of replacing the
    built-in environment.

    Args:
        module_path: Python module that defines the factory.
        class_name: Name of the factory class in that module.

    Returns:
        The factory class registered under its ``name``.
    """
    ensure_environments_registered()
    environment_factory_type = getattr(importlib.import_module(module_path), class_name)
    factory_class_path = f"{module_path}.{class_name}"
    environment_name = getattr(environment_factory_type, "name", None)
    environment_registry = EnvironmentRegistry()
    assert environment_name is not None and environment_registry.is_registered(
        environment_name
    ), f"Environment factory {factory_class_path} is not registered. Decorate it with @register_environment."
    registered_factory_type = environment_registry.get_component_by_name(environment_name)
    assert registered_factory_type is environment_factory_type, (
        f"Environment factory {factory_class_path} uses the name '{environment_name}', which is already registered"
        f" by {registered_factory_type.__module__}.{registered_factory_type.__qualname__}. Choose a unique name."
    )
    return environment_factory_type


# Legacy argparse compatibility
#
# Environment factories expose only build(cfg). Until the argparse frontend is
# retired, this section generates their environment-subcommand flags from the config
# registered with EnvironmentRegistry and reconstructs that config from the resulting
# Namespace.
# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this section when runners receive
# typed configs directly.
_FIELDS_PROVIDED_BY_SHARED_PARSERS = {"auto", "enable_cameras", "mimic", "num_envs"}


def add_environment_cli_args(
    parser: argparse.ArgumentParser,
    environment_factory_type: type[ArenaEnvironmentFactory],
) -> None:
    """Add legacy CLI flags generated from an environment factory's registered config."""
    environment_cfg_type = EnvironmentRegistry().get_environment_cfg_type(environment_factory_type)
    add_dataclass_cli_args(
        parser,
        environment_cfg_type,
        excluded_fields=_FIELDS_PROVIDED_BY_SHARED_PARSERS,
    )

    add_cli_only_args = getattr(environment_factory_type, "_add_legacy_cli_only_args", None)
    if add_cli_only_args is not None:
        add_cli_only_args(parser)


def _environment_cfg_from_cli(
    environment_factory_type: type[ArenaEnvironmentFactory],
    args_cli: argparse.Namespace,
) -> ArenaEnvironmentCfg:
    """Create a typed environment config from matching legacy Namespace values."""
    environment_cfg_type = EnvironmentRegistry().get_environment_cfg_type(environment_factory_type)
    return dataclass_from_cli(environment_cfg_type, args_cli)


def build_environment_from_cli(
    environment_factory_type: type[ArenaEnvironmentFactory],
    args_cli: argparse.Namespace,
) -> IsaacLabArenaEnvironment:
    """Build an environment from the typed config reconstructed from legacy CLI values."""
    environment_factory = environment_factory_type()
    return environment_factory.build(_environment_cfg_from_cli(environment_factory_type, args_cli))


def import_external_environment(environment_class_path: str) -> type[ArenaEnvironmentFactory]:
    """Import the environment factory selected by ``--external_environment_class_path``.

    Importing the module registers the factory through its ``@register_environment``
    decorator. Repeated imports return the same registered factory.

    Args:
        environment_class_path: Factory location in the form ``module_path:ClassName``.

    Returns:
        The registered environment factory class.
    """
    if ":" not in environment_class_path:
        raise ValueError(
            f"Invalid environment path: {environment_class_path}. Expected format: 'module_path:class_name'"
        )
    module_path, class_name = environment_class_path.split(":", 1)
    try:
        return import_registered_environment_factory(module_path, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"Could not resolve the environment path '{environment_class_path}' into an environment class."
            " The format should be 'module_path:class_name'.\n"
            f"Received the error:\n {e}."
        ) from e


def add_cli_override_args(parser: argparse.ArgumentParser, override_specs: list[CliOverrideSpec]) -> None:
    """Add each declared override to the CLI ``parser`` as a ``--flag``."""
    for override in override_specs:
        flag = f"--{override.arg}"
        assert flag not in parser._option_string_actions, (  # noqa: SLF001
            f"CLI override flag '{flag}' (asset '{override.target_node_id}') is already a parser flag "
            "(e.g. --num_envs/--seed or an AppLauncher flag); rename its 'arg' in the YAML."
        )
        parser.add_argument(
            flag,
            type=str,
            default=None,
            help=f"Override the registry name behind graph asset '{override.target_node_id}'.",
        )


def add_example_environments_cli_args(args_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()

    args, unknown = args_parser.parse_known_args()
    environment = getattr(args, "external_environment_class_path", None)
    if environment is not None:
        print(f"Adding external environment: {environment}")
        import_external_environment(environment)

    # A graph spec YAML may declare its own swappable flags under `cli_override_specs`. Register them
    # here, before parsing, so they appear in --help and parse like any other flag.
    env_spec = getattr(args, "env_spec", None)
    if env_spec is not None:
        # The env comes from the graph spec, so don't register the example-environment subparsers.
        cli_override_specs_from_yaml = ArenaEnvGraphSpec.read_cli_override_specs(env_spec)
        add_cli_override_args(args_parser, cli_override_specs_from_yaml)
        return args_parser

    subparsers = args_parser.add_subparsers(
        dest="example_environment", required=False, help="Example environment to run"
    )
    for env_name in env_registry.get_all_keys():
        environment_factory_type = env_registry.get_component_by_name(env_name)
        subparser = subparsers.add_parser(environment_factory_type.name)
        add_environment_cli_args(subparser, environment_factory_type)

    return args_parser


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this environment-subparser pipeline after
# runner scripts receive typed environment configs.
def get_isaaclab_arena_environments_cli_parser(
    args_parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if args_parser is None:
        args_parser = get_isaaclab_arena_cli_parser()
    # NOTE(alexmillane, 2025.09.04): This command adds subparsers for each example environment.
    # So it has to be added last, because the subparser flags are parsed after the others.
    args_parser = add_example_environments_cli_args(args_parser)
    return args_parser


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this construction pipeline after experiment_runner,
# policy_runner, imitation-learning scripts, and notebooks pass typed environment and
# builder configs instead of an argparse Namespace.
def get_arena_builder_from_cli(
    args_cli: argparse.Namespace,
    hydra_overrides: list[str] | None = None,
) -> ArenaEnvBuilder:
    """Build an :class:`ArenaEnvBuilder` from parsed CLI args.

    Args:
        args_cli: Parsed argparse namespace.
        hydra_overrides: Optional Hydra variation override strings (e.g.
            ``"light.hdr_image.enabled=true"``).
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # The env comes from exactly one source: a graph spec YAML (--env_spec) or a
    # registered example-environment name (subcommand).
    env_spec = getattr(args_cli, "env_spec", None)
    example_environment = getattr(args_cli, "example_environment", None)
    assert (env_spec is None) != (example_environment is None), (
        "Specify exactly one environment source: an example-environment name or --env_spec"
        f" (got example_environment={example_environment!r}, env_spec={env_spec!r})"
    )

    # Either env graph spec yaml OR example env name
    arena_env = (
        arena_env_from_graph_spec(env_spec, args_cli)
        if env_spec is not None
        else _arena_env_from_example_name(example_environment, args_cli)
    )
    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    return ArenaEnvBuilder(arena_env, builder_cfg, hydra_overrides=hydra_overrides)


def arena_env_from_graph_spec(env_spec: str, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
    """Build the arena env from a graph spec YAML, applying any CLI node overrides."""
    # Graph specs may reference components registered by extension packages, so register first.
    ensure_environments_registered()
    spec = ArenaEnvGraphSpec.from_yaml(env_spec)
    spec.apply_cli_override_args(args_cli)
    # cameras are enabled in embodiment, need to pass along to the env
    return spec.to_arena_env(enable_cameras=args_cli.enable_cameras)


def _arena_env_from_example_name(example_environment: str, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
    """Build the arena env from a registered example-environment name (subcommand)."""
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    assert env_registry.is_registered(
        example_environment
    ), f"Example environment type {example_environment} not supported"
    environment_factory_type = env_registry.get_component_by_name(example_environment)
    return build_environment_from_cli(environment_factory_type, args_cli)
