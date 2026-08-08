# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep graph-YAML evaluation environments on their temporary argparse path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.legacy_environment_cli_args import legacy_environment_args_to_cli_args
from isaaclab_arena_environments.cli import arena_env_from_graph_spec, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

# TODO(cvolk, 2026-07-07): [typed-config-migration] Delete this module when graph-YAML environments have a
# typed configuration and factory. Until then, only graph construction crosses the
# argparse compatibility boundary; policy, rollout, and rebuild execution stay typed.


@dataclass
class LegacyGraphEnvironmentCfg(ArenaEnvironmentCfg):
    """Environment config for graph-YAML environments

    The environment is stored as env_spec_path and the per-run overrides.
    """

    env_spec_path: str = ""
    """Graph-spec YAML path the environment was loaded from; serialized as the environment ``type``."""

    per_run_overrides: dict[str, Any] = field(default_factory=dict)
    """The Run's ``environment`` YAML values minus the environment path itself
    i.e. the per-run overrides e.g. {"pick_up_object": "banana"}. Combined with the path to
    re-serialize the run and build the graph-environment CLI tokens at execution time.
    """


def build_arena_builder_from_legacy_graph(
    cfg: LegacyGraphEnvironmentCfg,
    environment_builder: ArenaEnvBuilderCfg,
    hydra_overrides: list[str],
) -> ArenaEnvBuilder:
    """Build a graph-YAML environment through the existing argparse adapter."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    assert cfg.env_spec_path.endswith((".yaml", ".yml")), "legacy graph config must select a graph YAML"
    arena_env_args = legacy_environment_args_to_cli_args({"environment": cfg.env_spec_path, **cfg.per_run_overrides})
    parser = get_isaaclab_arena_environments_cli_parser()
    args_cli = parser.parse_args(arena_env_args)
    arena_env = arena_env_from_graph_spec(args_cli.env_spec, args_cli)
    return ArenaEnvBuilder(arena_env, environment_builder, hydra_overrides=hydra_overrides)
