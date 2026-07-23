# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep graph-YAML evaluation environments on their temporary argparse path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena_environments.cli import arena_env_from_graph_spec, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

# TODO(cvolk, 2026-07-07): [typed-config-migration] Delete this module when graph-YAML environments have a
# typed configuration and factory. Until then, only graph construction crosses the
# argparse compatibility boundary; policy, rollout, and rebuild execution stay typed.


@dataclass
class LegacyGraphEnvironmentCfg(ArenaEnvironmentCfg):
    """Carry a graph-YAML environment through its temporary CLI construction path."""

    arena_env_args: list[str]
    """Arguments consumed by the existing graph-environment parser."""

    env_graph_spec_yaml: str = ""
    """Graph-spec YAML path the environment was loaded from."""

    environment_values: dict[str, Any] = field(default_factory=dict)
    """Environment values (without the type selector) used to re-serialize the Run."""


def build_arena_builder_from_legacy_graph(
    cfg: LegacyGraphEnvironmentCfg,
    environment_builder: ArenaEnvBuilderCfg,
    hydra_overrides: list[str],
) -> ArenaEnvBuilder:
    """Build a graph-YAML environment through the existing argparse adapter.

    Only environment construction crosses the argparse boundary; the Run's typed
    builder configuration is used directly, so Hydra overrides on it take effect.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    assert "--env_graph_spec_yaml" in cfg.arena_env_args, "legacy graph config must select a graph YAML"
    parser = get_isaaclab_arena_environments_cli_parser()
    args_cli = parser.parse_args(cfg.arena_env_args)
    arena_env = arena_env_from_graph_spec(args_cli.env_graph_spec_yaml, args_cli)
    return ArenaEnvBuilder(arena_env, environment_builder, hydra_overrides=hydra_overrides)
