# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute typed Arena experiments independently of their configuration frontend."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, ArenaExperimentResult, ExperimentStatus
from isaaclab_arena.evaluation.rollout import rollout_policy
from isaaclab_arena.metrics.aggregate_metrics import aggregate_metrics
from isaaclab_arena.utils.hydra_overrides import hydra_overrides_from_nested_dict
from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
    collect_garbage_and_clear_cuda_cache,
    teardown_simulation_app,
)
from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

ArenaBuilderFactory = Callable[[ArenaExperimentCfg], "ArenaEnvBuilder"]


def build_arena_builder(cfg: ArenaExperimentCfg) -> ArenaEnvBuilder:
    """Build an Arena compiler from a registered environment configuration."""
    # ArenaEnvBuilder imports Isaac Lab runtime modules and must be loaded only after
    # the dispatcher has started SimulationApp.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    environment_factory_type = EnvironmentRegistry().get_factory_type_for_cfg(cfg.environment)
    arena_environment = environment_factory_type().build(cfg.environment)

    return ArenaEnvBuilder(
        arena_environment,
        cfg.environment_builder,
        hydra_overrides=hydra_overrides_from_nested_dict(cfg.variations),
    )


def run_experiment(
    cfg: ArenaExperimentCfg,
    *,
    output_dir: str | Path,
    video_cfg: VideoRecordingCfg | None = None,
    arena_builder_factory: ArenaBuilderFactory = build_arena_builder,
) -> ArenaExperimentResult:
    """Execute one typed experiment and return its result."""
    # TODO(cvolk, 2026-07-06): Remove the injectable legacy builder path when
    # environment graphs support typed construction outside their argparse frontend.
    started_at = time.time()
    metrics_per_rebuild: list[MetricsDataCollection] = []
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
                env = _make_environment(cfg, rebuild_video_cfg.render_mode, arena_builder_factory)
                results_path = os.path.join(output_dir, f"episode_results_rebuild{rebuild_index}.jsonl")
                env.unwrapped.episode_recorder.set_job_name(cfg.name)
                env.unwrapped.episode_recorder.set_output_path(results_path)

                policy = _make_policy(cfg)
                num_steps, num_episodes = _resolve_rollout_limit(cfg, policy, num_episodes)
                env = wrap_env_for_video(env, rebuild_video_cfg, num_steps, num_episodes)
                metrics = rollout_policy(env, policy, num_steps=num_steps, num_episodes=num_episodes)
                if metrics is not None:
                    metrics_per_rebuild.append(metrics)
            finally:
                _close_experiment_resources(policy, env)

    except Exception:  # noqa: BLE001 -- failures are returned for dispatcher policy
        return ArenaExperimentResult(
            experiment_name=cfg.name,
            status=ExperimentStatus.FAILED,
            started_at=started_at,
            ended_at=time.time(),
            metrics=_aggregate_metrics(metrics_per_rebuild),
            error=traceback.format_exc(),
        )

    return ArenaExperimentResult(
        experiment_name=cfg.name,
        status=ExperimentStatus.COMPLETED,
        started_at=started_at,
        ended_at=time.time(),
        metrics=_aggregate_metrics(metrics_per_rebuild),
    )


def _make_environment(
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


def _make_policy(cfg: ArenaExperimentCfg) -> PolicyBase:
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
) -> tuple[int | None, int | None]:
    """Resolve explicit rollout limits or a replay policy's intrinsic length."""
    num_steps = cfg.rollout.num_steps
    if num_steps is None and num_episodes is None:
        assert policy.has_length(), f"Experiment '{cfg.name}' must configure num_steps or num_episodes"
        num_steps = policy.length()
        assert num_steps is not None and num_steps > 0, f"Policy for experiment '{cfg.name}' has no usable length"
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


def _aggregate_metrics(metrics_per_rebuild: list[MetricsDataCollection]) -> MetricsDataCollection | None:
    """Aggregate available rebuild metrics, or return ``None`` when a task defines none."""
    return aggregate_metrics(metrics_per_rebuild) if metrics_per_rebuild else None


def _close_policy(policy: PolicyBase | None) -> None:
    try:
        if policy is not None:
            policy.close()
    finally:
        collect_garbage_and_clear_cuda_cache()


def _close_environment(env: gym.Env | None) -> None:
    if env is None:
        return
    try:
        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
    finally:
        try:
            env.close()
        finally:
            collect_garbage_and_clear_cuda_cache()


def _close_experiment_resources(policy: PolicyBase | None, env: gym.Env | None) -> None:
    try:
        _close_policy(policy)
    finally:
        _close_environment(env)
