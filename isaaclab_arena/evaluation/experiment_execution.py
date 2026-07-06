# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and execute typed Arena experiments for an evaluation frontend."""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, ArenaExperimentResult, ExperimentStatus
from isaaclab_arena.evaluation.policy_runner import rollout_policy
from isaaclab_arena.evaluation.resource_cleanup import close_experiment_resources
from isaaclab_arena.metrics.aggregate_metrics import aggregate_metrics
from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

ArenaBuilderFactory: TypeAlias = Callable[[ArenaExperimentCfg], "ArenaEnvBuilder"]


def build_and_run_experiment(
    cfg: ArenaExperimentCfg,
    output_dir: str | Path,
    arena_builder_factory: ArenaBuilderFactory,
    video_cfg: VideoRecordingCfg | None = None,
    fallback_num_steps: int | None = None,
) -> ArenaExperimentResult:
    """Build and run one typed experiment, then return its result."""
    started_at = time.time()
    metrics_per_rebuild: list[MetricsDataCollection] = []
    status = ExperimentStatus.COMPLETED
    error = None
    output_dir = str(output_dir)
    video_cfg = video_cfg or VideoRecordingCfg(video_base_dir=output_dir)
    episodes_per_rebuild = _split_episodes_across_rebuilds(
        cfg.rollout.num_episodes,
        cfg.num_rebuilds,
        cfg.name,
    )

    try:
        for rebuild_index, num_episodes in enumerate(episodes_per_rebuild):
            env = None
            policy = None
            try:
                rebuild_video_cfg = replace(
                    video_cfg,
                    video_base_dir=output_dir,
                    camera_name_prefix=f"robot-cam-rebuild{rebuild_index}",
                )
                env = _build_environment(cfg, rebuild_video_cfg.render_mode, arena_builder_factory)
                results_path = os.path.join(output_dir, f"episode_results_rebuild{rebuild_index}.jsonl")
                env.unwrapped.episode_recorder.set_job_name(cfg.name)
                env.unwrapped.episode_recorder.set_output_path(results_path)

                policy = _build_policy(cfg)
                num_steps, num_episodes = _resolve_rollout_limit(
                    cfg,
                    policy,
                    num_episodes,
                    fallback_num_steps,
                )
                env = wrap_env_for_video(env, rebuild_video_cfg, num_steps, num_episodes)
                metrics = rollout_policy(env, policy, num_steps=num_steps, num_episodes=num_episodes)
                if metrics is not None:
                    metrics_per_rebuild.append(metrics)
            finally:
                close_experiment_resources(policy, env)

    except Exception:  # noqa: BLE001 -- failures are returned for dispatcher policy
        status = ExperimentStatus.FAILED
        error = traceback.format_exc()

    return ArenaExperimentResult(
        experiment_name=cfg.name,
        status=status,
        started_at=started_at,
        ended_at=time.time(),
        metrics=aggregate_metrics(metrics_per_rebuild) if metrics_per_rebuild else None,
        error=error,
    )


def _build_environment(
    cfg: ArenaExperimentCfg,
    render_mode: str | None,
    arena_builder_factory: ArenaBuilderFactory,
) -> gym.Env:
    """Compile and instantiate an experiment's environment."""
    arena_builder = arena_builder_factory(cfg)
    _, env_cfg, env_kwargs = arena_builder.build_registered()
    if env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{cfg.name}"
    return arena_builder.make_registered(env_cfg, env_kwargs, render_mode=render_mode)


def _build_policy(cfg: ArenaExperimentCfg) -> PolicyBase:
    """Instantiate the registered policy configured by an experiment."""
    policy_cfg = _policy_cfg_for_num_envs(cfg.policy, cfg.environment_builder.num_envs)
    policy_type = PolicyRegistry().get_policy_type_for_cfg(policy_cfg)
    return policy_type(policy_cfg)


def _policy_cfg_for_num_envs(policy_cfg: PolicyCfg, num_envs: int) -> PolicyCfg:
    """Align legacy policy batch-size fields with the environment batch size."""
    # TODO(cvolk, 2026-07-06): Remove the duplicated policy ``num_envs`` fields and
    # this adapter once policies receive the environment batch size as runtime context.
    if "num_envs" not in {config_field.name for config_field in fields(policy_cfg)}:
        return policy_cfg
    return replace(policy_cfg, num_envs=num_envs)


def _resolve_rollout_limit(
    cfg: ArenaExperimentCfg,
    policy: PolicyBase,
    num_episodes: int | None,
    fallback_num_steps: int | None,
) -> tuple[int | None, int | None]:
    """Resolve explicit rollout limits or a replay policy's intrinsic length."""
    num_steps = cfg.rollout.num_steps
    if num_steps is None and num_episodes is None:
        if policy.has_length():
            num_steps = policy.length()
            assert num_steps is not None and num_steps > 0, f"Policy for experiment '{cfg.name}' has no usable length"
        else:
            # TODO(cvolk, 2026-07-06): Remove this fallback when the legacy eval-runner
            # CLI no longer supplies a process-wide default rollout length.
            assert (
                fallback_num_steps is not None and fallback_num_steps > 0
            ), f"Experiment '{cfg.name}' must configure num_steps or num_episodes"
            num_steps = fallback_num_steps
    return num_steps, num_episodes


def _split_episodes_across_rebuilds(
    num_episodes: int | None,
    num_rebuilds: int,
    experiment_name: str,
) -> list[int | None]:
    """Split an experiment's episode budget as evenly as possible across rebuilds."""
    if num_episodes is None:
        return [None] * num_rebuilds
    assert num_episodes >= num_rebuilds, (
        f"Experiment '{experiment_name}': num_episodes ({num_episodes}) must be >= num_rebuilds "
        f"({num_rebuilds}) so each rebuild runs at least one episode"
    )
    base, remainder = divmod(num_episodes, num_rebuilds)
    return [base + int(rebuild_index < remainder) for rebuild_index in range(num_rebuilds)]
