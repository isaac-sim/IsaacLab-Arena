# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep graph-YAML evaluation environments on their temporary argparse path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena_environments.cli import (
    arena_env_from_graph_spec,
    get_isaaclab_arena_environments_cli_parser,
    load_graph_spec_with_cli_overrides,
)

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.policy.policy_base import PolicyCfg

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
    rebuild_index: int = 0,
    policy_cfg: PolicyCfg | None = None,
) -> ArenaEnvBuilder:
    """Build a graph-YAML environment through the existing argparse adapter.

    Args:
        cfg: Graph-spec environment selection and per-run overrides.
        environment_builder: Builder options for the environment.
        hydra_overrides: Variation overrides applied when the environment config is composed.
        policy_cfg: Config of the policy that will drive the environment, used by
            ``camera_resolution_mode: policy_default`` to fit the embodiment cameras to the policy.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = _parse_graph_environment_cli_args(cfg)
    arena_env = arena_env_from_graph_spec(args_cli.env_spec, args_cli)
    return ArenaEnvBuilder(
        arena_env,
        environment_builder,
        hydra_overrides=hydra_overrides,
        rebuild_index=rebuild_index,
        policy_cfg=policy_cfg,
    )


def load_graph_spec_from_legacy_graph(cfg: LegacyGraphEnvironmentCfg) -> ArenaEnvGraphSpec:
    """Load a Run's graph spec with its per-run overrides applied, as the environment is built from it."""
    args_cli = _parse_graph_environment_cli_args(cfg)
    return load_graph_spec_with_cli_overrides(args_cli.env_spec, args_cli)


def _parse_graph_environment_cli_args(cfg: LegacyGraphEnvironmentCfg) -> argparse.Namespace:
    """Parse the command-line arguments that build this graph-spec environment."""
    assert cfg.env_spec_path.endswith((".yaml", ".yml")), "legacy graph config must select a graph YAML"
    parser = get_isaaclab_arena_environments_cli_parser()
    return parser.parse_args(_graph_environment_cli_args(cfg))


def _graph_environment_cli_args(cfg: LegacyGraphEnvironmentCfg) -> list[str]:
    """Return the command-line arguments that build this graph-spec environment.

    For example, per-run overrides ``{"enable_cameras": True, "object": "box", "mimic": False}`` give
    ``["--env_spec", <env_spec_path>, "--enable_cameras", "--object", "box"]``.
    """
    cli_args = ["--env_spec", cfg.env_spec_path]
    for name, value in cfg.per_run_overrides.items():
        if value is None or value is False:
            # An unset value or a switched-off flag is not passed at all.
            continue
        if value is True:
            # Boolean flags such as --enable_cameras take no value.
            cli_args.append(f"--{name}")
        else:
            cli_args.extend([f"--{name}", str(value)])
    return cli_args
