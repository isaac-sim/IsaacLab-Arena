# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve rollout limits for recorded episode-condition replay."""

from __future__ import annotations

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.variations.episode_conditions import load_episode_conditions_overlay


def replay_episode_count(builder_cfg: ArenaEnvBuilderCfg) -> int | None:
    """Return the source-row count when condition replay is enabled."""
    if builder_cfg.episode_conditions_path is None:
        return None
    count = load_episode_conditions_overlay(builder_cfg.episode_conditions_path).num_conditions
    assert count > 0, "Rebuild conditions must list at least one episode"
    return count


def resolve_replay_episode_budget(
    builder_cfg: ArenaEnvBuilderCfg,
    explicit_num_episodes: int | None,
) -> int | None:
    """Use an explicit episode budget or default to one exact pass."""
    source_count = replay_episode_count(builder_cfg)
    if source_count is None:
        return explicit_num_episodes
    return source_count if explicit_num_episodes is None else explicit_num_episodes


def assert_replay_compatible_run_cfg(cfg: ArenaRunCfg) -> None:
    """Reject experiment-runner settings incompatible with condition replay."""
    if cfg.environment_builder.episode_conditions_path is None:
        return
    assert cfg.num_rebuilds == 1, f"Run '{cfg.name}' sets episode_conditions_path; num_rebuilds must be 1."
    assert (
        cfg.rollout_limit.num_steps is None
    ), f"Run '{cfg.name}' replays episode conditions; num_steps is not supported."


def resolve_policy_runner_replay_budget(
    builder_cfg: ArenaEnvBuilderCfg,
    *,
    num_steps: int | None,
    num_episodes: int | None,
) -> int | None:
    """Validate policy-runner limits and return its episode budget."""
    if builder_cfg.episode_conditions_path is None:
        return num_episodes
    assert num_steps is None, "episode_conditions_path replay does not support --num_steps"
    return resolve_replay_episode_budget(builder_cfg, num_episodes)
