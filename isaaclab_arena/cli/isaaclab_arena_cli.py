# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.dataclass_cli import (
    add_dataclass_cli_args,
    assert_cli_defaults_match_dataclass,
    dataclass_from_cli,
)
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

# This set changes only where flags appear in --help; their definitions still come
# from ArenaEnvBuilderCfg.
# TODO(cvolk, 2026-07-06): Remove this legacy help-group mapping when nested builder
# configs introduce responsibility-based CLI groups.
_LEGACY_ISAAC_LAB_BUILDER_CFG_FIELDS = frozenset({"disable_fabric", "env_spacing", "mimic", "num_envs", "seed"})


# TODO(cvolk, 2026-07-03): Delete this Namespace-to-config adapter after policy_runner,
# eval_runner, and the remaining argparse scripts pass ArenaEnvBuilderCfg directly.
def arena_env_builder_cfg_from_argparse(args_cli: argparse.Namespace) -> ArenaEnvBuilderCfg:
    """Translate parsed CLI arguments into the typed builder configuration.

    Args:
        args_cli: Parsed Arena and Isaac Lab command-line arguments.

    Returns:
        The configuration consumed by ``ArenaEnvBuilder``.
    """
    return dataclass_from_cli(ArenaEnvBuilderCfg, args_cli)


# TODO(cvolk, 2026-07-03): Delete this parser pipeline and its add_* helpers after
# policy_runner, eval_runner, and the remaining argparse scripts accept typed configs.
def get_isaaclab_arena_cli_parser() -> argparse.ArgumentParser:
    """Get a complete argument parser with both Isaac Lab and IsaacLab Arena arguments."""
    parser = argparse.ArgumentParser(description="IsaacLab Arena CLI parser.")
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)
    add_external_environments_cli_args(parser)
    add_env_graph_spec_cli_args(parser)
    return parser


def add_isaac_lab_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add builder and distributed flags historically grouped under Isaac Lab."""

    isaac_lab_group = parser.add_argument_group("Isaac Lab Arguments", "Arguments specific to Isaac Lab framework")
    add_dataclass_cli_args(
        isaac_lab_group,
        ArenaEnvBuilderCfg,
        included_fields=_LEGACY_ISAAC_LAB_BUILDER_CFG_FIELDS,
    )
    # TODO(cvolk, 2026-07-06): Move --distributed into a typed runner or simulation-app
    # config. It controls AppLauncher and policy_runner process setup, not ArenaEnvBuilder.
    isaac_lab_group.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Run distributed (one process per GPU). Use with torchrun; AppLauncher uses LOCAL_RANK for device.",
    )


def add_isaaclab_arena_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add Isaac Lab Arena specific command line arguments to the given parser."""
    arena_group = parser.add_argument_group(
        "Isaac Lab Arena Arguments", "Arguments specific to Isaac Lab Arena framework"
    )

    # AppLauncher already owns --device. Verify that its default agrees with the
    # builder config before generating the remaining builder flags.
    assert_cli_defaults_match_dataclass(parser, ArenaEnvBuilderCfg, {"device"})
    # TODO(cvolk, 2026-07-06): Delete this custom spelling when the argparse builder
    # adapter is removed. It exists only to preserve the current CLI during migration.
    arena_group.add_argument(
        "--no-solve-relations",
        action="store_false",
        dest="solve_relations",
        default=ArenaEnvBuilderCfg().solve_relations,
        help="Disable solving spatial relations in the environment.",
    )
    add_dataclass_cli_args(
        arena_group,
        ArenaEnvBuilderCfg,
        # Preserve Arena's existing negative-only --no-solve-relations flag.
        # Generic bool handling would also add --solve_relations and would spell
        # the negative flag --no-solve_relations.
        # AppLauncher owns --device, while policy_runner preserves the historical
        # location of --language_instruction.
        excluded_fields={"device", "language_instruction", "solve_relations"} | _LEGACY_ISAAC_LAB_BUILDER_CFG_FIELDS,
    )
    arena_group.add_argument(
        "--list-variations",
        action="store_true",
        default=False,
        help="Print Hydra-configurable variations for the selected environment and exit.",
    )


def add_env_graph_spec_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add environment graph spec specific command line arguments to the given parser."""
    env_graph_spec_group = parser.add_argument_group(
        "Environment Graph Spec Arguments", "Arguments specific to environment graph spec"
    )
    env_graph_spec_group.add_argument(
        "--env_graph_spec_yaml",
        type=str,
        default=None,
        help=(
            "Path to an environment graph spec YAML. When set, the environment is built from the graph spec instead of"
            " a registered example-environment name; the env-name subcommand then becomes optional. Any override flags"
            " the YAML declares under `cli_override_specs` are added to the parser dynamically."
        ),
    )


def add_external_environments_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add external environments specific command line arguments to the given parser."""
    external_environments_group = parser.add_argument_group(
        "External Environments Arguments", "Arguments specific to external environments"
    )
    external_environments_group.add_argument(
        "--external_environment_class_path",
        type=str,
        default=None,
        help="Name of the external environment to run",
    )
