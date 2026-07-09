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
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

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


# Legacy argparse compatibility
#
# First-party factories now expose only build(cfg). Until the argparse frontend is
# retired, this section generates their environment-subcommand flags and reconstructs
# the same typed config from the resulting Namespace. External factories that still
# inherit ExampleEnvironmentBase continue using their own add_cli_args() and get_env().
# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this section and the factories'
# _legacy_argparse_cfg_type declarations when runners receive typed configs directly.
_FIELDS_PROVIDED_BY_SHARED_PARSERS = {"auto", "enable_cameras", "mimic", "num_envs"}


def _get_legacy_argparse_cfg_type(
    environment_factory_type: type[ArenaEnvironmentFactory],
) -> type[ArenaEnvironmentCfg]:
    """Return the config dataclass used by a first-party factory's legacy CLI adapter."""
    environment_cfg_type = getattr(environment_factory_type, "_legacy_argparse_cfg_type", None)
    assert (
        environment_cfg_type is not None
    ), f"{environment_factory_type.__name__} must define _legacy_argparse_cfg_type while argparse is supported"
    return environment_cfg_type


def add_environment_cli_args(
    parser: argparse.ArgumentParser,
    environment_factory_type: type[ArenaEnvironmentFactory],
) -> None:
    """Add legacy CLI flags for a typed or external environment factory."""
    if issubclass(environment_factory_type, ExampleEnvironmentBase):
        environment_factory_type.add_cli_args(parser)
        return

    environment_cfg_type = _get_legacy_argparse_cfg_type(environment_factory_type)
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
    environment_cfg_type = _get_legacy_argparse_cfg_type(environment_factory_type)
    return dataclass_from_cli(environment_cfg_type, args_cli)


def build_environment_from_cli(
    environment_factory_type: type[ArenaEnvironmentFactory],
    args_cli: argparse.Namespace,
) -> IsaacLabArenaEnvironment:
    """Build an environment through its typed or external legacy CLI path."""
    environment_factory = environment_factory_type()
    if isinstance(environment_factory, ExampleEnvironmentBase):
        return environment_factory.get_env(args_cli)
    return environment_factory.build(_environment_cfg_from_cli(environment_factory_type, args_cli))


def parse_and_return_external_environment_from_string(
    environment_path: str,
) -> tuple[str, type[ArenaEnvironmentFactory]]:
    """Parse a string and import the environment class

    Args:
        environment_path: The path to the environment class

    Raises:
        ValueError: If the environment path is not in the format "module_path:class_name"

    Returns:
        A tuple containing the environment name and factory class.
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
        name, cls = parse_and_return_external_environment_from_string(environment)
        env_registry.register(cls, name)

    # A graph spec YAML may declare its own swappable flags under `cli_override_specs`. Register them
    # here, before parsing, so they appear in --help and parse like any other flag.
    env_graph_spec_yaml = getattr(args, "env_graph_spec_yaml", None)
    if env_graph_spec_yaml is not None:
        # The env comes from the graph spec, so don't register the example-environment subparsers.
        cli_override_specs_from_yaml = ArenaEnvGraphSpec.read_cli_override_specs(env_graph_spec_yaml)
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


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this environment-subparser pipeline with the
# per-environment add_cli_args() and get_env(args_cli) adapters after runner scripts
# receive typed environment configs.
def get_isaaclab_arena_environments_cli_parser(
    args_parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if args_parser is None:
        args_parser = get_isaaclab_arena_cli_parser()
    # NOTE(alexmillane, 2025.09.04): This command adds subparsers for each example environment.
    # So it has to be added last, because the subparser flags are parsed after the others.
    args_parser = add_example_environments_cli_args(args_parser)
    return args_parser


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this construction pipeline after eval_runner,
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

    # The env comes from exactly one source: a graph spec YAML (--env_graph_spec_yaml) or a
    # registered example-environment name (subcommand).
    env_graph_spec_yaml = getattr(args_cli, "env_graph_spec_yaml", None)
    example_environment = getattr(args_cli, "example_environment", None)
    assert (env_graph_spec_yaml is None) != (example_environment is None), (
        "Specify exactly one environment source: an example-environment name or --env_graph_spec_yaml"
        f" (got example_environment={example_environment!r}, env_graph_spec_yaml={env_graph_spec_yaml!r})"
    )

    # Either env graph spec yaml OR example env name
    arena_env = (
        _arena_env_from_graph_spec(env_graph_spec_yaml, args_cli)
        if env_graph_spec_yaml is not None
        else _arena_env_from_example_name(example_environment, args_cli)
    )
    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    return ArenaEnvBuilder(arena_env, builder_cfg, hydra_overrides=hydra_overrides)


def _arena_env_from_graph_spec(env_graph_spec_yaml: str, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
    """Build the arena env from a graph spec YAML, applying any CLI node overrides."""
    spec = ArenaEnvGraphSpec.from_yaml(env_graph_spec_yaml)
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
