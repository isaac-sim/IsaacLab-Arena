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
    from isaaclab_arena.policy.policy_base import PolicyCfg

# TODO(cvolk, 2026-07-07): [typed-config-migration] Delete this module when graph-YAML environments have a
# typed configuration and factory. Until then, only graph construction crosses the
# argparse compatibility boundary; policy, rollout, and rebuild execution stay typed.


@dataclass
class LegacyGraphEnvironmentCfg(ArenaEnvironmentCfg):
    """Graph-YAML selection, asset overrides, and per-Run Isaac Lab configuration."""

    env_spec_path: str = ""
    """Graph-spec YAML path the environment was loaded from; serialized as the environment ``type``."""

    per_run_overrides: dict[str, Any] = field(default_factory=dict)
    """Graph CLI values from the Run's ``environment`` mapping, such as ``pick_up_object``.

    Excludes ``type`` and ``env_cfg_override``, which have dedicated fields.
    """

    env_cfg_override: dict[str, Any] | None = None
    """Validated Isaac Lab config overrides applied after the graph's environment callback."""


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

    assert cfg.env_spec_path.endswith((".yaml", ".yml")), "legacy graph config must select a graph YAML"
    arena_env_args = legacy_environment_args_to_cli_args({"environment": cfg.env_spec_path, **cfg.per_run_overrides})
    parser = get_isaaclab_arena_environments_cli_parser()
    args_cli = parser.parse_args(arena_env_args)
    arena_env = arena_env_from_graph_spec(args_cli.env_spec, args_cli)
    if cfg.env_cfg_override is not None:
        from isaaclab_arena.environment_spec.env_cfg_override import apply_env_cfg_override

        graph_callback = arena_env.env_cfg_callback

        def apply_run_env_cfg_override(env_cfg):
            if graph_callback is not None:
                env_cfg = graph_callback(env_cfg)
            return apply_env_cfg_override(env_cfg, cfg.env_cfg_override)

        arena_env.env_cfg_callback = apply_run_env_cfg_override
    return ArenaEnvBuilder(
        arena_env,
        environment_builder,
        hydra_overrides=hydra_overrides,
        rebuild_index=rebuild_index,
        policy_cfg=policy_cfg,
    )
