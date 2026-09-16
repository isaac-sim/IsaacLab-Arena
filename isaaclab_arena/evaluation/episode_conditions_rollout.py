# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rollout limits when replaying recorded variation conditions."""

from __future__ import annotations

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.variations.episode_conditions import load_episode_conditions_overlay


def replay_episode_count(builder_cfg: ArenaEnvBuilderCfg) -> int | None:
    """Return the fixed episode budget when ``episode_conditions_path`` is set."""
    if builder_cfg.episode_conditions_path is None:
        return None
    return load_episode_conditions_overlay(builder_cfg.episode_conditions_path).num_conditions


def assert_replay_compatible_builder_cfg(builder_cfg: ArenaEnvBuilderCfg) -> None:
    """Reject builder settings that conflict with condition replay."""
    if builder_cfg.episode_conditions_path is None:
        return
    assert replay_episode_count(builder_cfg) > 0, "Condition overlay must list at least one episode"


def assert_replay_compatible_run_cfg(cfg: ArenaRunCfg) -> None:
    """Reject run rollout limits that conflict with condition replay."""
    if cfg.environment_builder.episode_conditions_path is None:
        return
    assert (
        cfg.num_rebuilds == 1
    ), f"Run '{cfg.name}' sets episode_conditions_path; num_rebuilds must be 1 (one overlay per rebuild)."
    assert (
        cfg.rollout_limit.num_steps is None
    ), f"Run '{cfg.name}' replays variation conditions; num_steps is not supported."
    assert (
        cfg.rollout_limit.num_episodes is None
    ), f"Run '{cfg.name}' replays variation conditions; set episode budget via the overlay, not num_episodes."
    assert_replay_compatible_builder_cfg(cfg.environment_builder)


def assert_replay_compatible_policy_runner_limits(
    *,
    episode_conditions_path: str | None,
    num_steps: int | None,
    num_episodes: int | None,
) -> None:
    """Reject explicit rollout limits when policy_runner replays conditions."""
    if episode_conditions_path is None:
        return
    assert num_steps is None, "episode_conditions_path replay does not support --num_steps"
    assert num_episodes is None, "episode_conditions_path replay sets the episode budget from the overlay"
