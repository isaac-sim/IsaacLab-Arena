# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and execute typed Arena runs and experiments for an evaluation frontend."""

from __future__ import annotations

import os
import traceback
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RunStatus
from isaaclab_arena.evaluation.legacy_graph_environment_cli import (
    LegacyGraphEnvironmentCfg,
    build_arena_builder_from_legacy_graph,
)
from isaaclab_arena.evaluation.policy_runner import rollout_policy
from isaaclab_arena.evaluation.resource_cleanup import close_run_resources
from isaaclab_arena.metrics.aggregate_metrics import aggregate_metrics
from isaaclab_arena.variations.variations_hydra import overrides_from_dict
from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


def execute_experiment(
    run_cfgs: list[ArenaRunCfg],
    output_dir: Path,
    record_viewport_video: bool = False,
    record_camera_video: bool = False,
    continue_on_error: bool = False,
) -> list[ArenaRunResult]:
    """Execute an experiment's runs in order and return their results.

    Args:
        run_cfgs: Ordered run configurations that make up the experiment.
        output_dir: Directory containing one output subdirectory per run.
        record_viewport_video: Whether to record the viewport for each run.
        record_camera_video: Whether to record observation cameras for each run.
        continue_on_error: Whether to continue with later runs after one fails.

    Returns:
        One result per attempted run, in execution order.
    """
    results = []
    for run_cfg in run_cfgs:
        print(f"Running run '{run_cfg.name}'", flush=True)
        run_output_dir = output_dir / run_cfg.name
        try:
            result = build_and_run(
                run_cfg,
                output_dir=run_output_dir,
                video_cfg=VideoRecordingCfg(
                    record_viewport_video=record_viewport_video,
                    record_camera_video=record_camera_video,
                    video_base_dir=str(run_output_dir),
                ),
            )
        except Exception as error:
            results.append(ArenaRunResult(run_name=run_cfg.name, status=RunStatus.FAILED))
            print(f"Run '{run_cfg.name}' failed with error: {error}", flush=True)
            print(f"Traceback: {traceback.format_exc()}", flush=True)
            if not continue_on_error:
                raise
            continue

        results.append(result)
    return results


def build_and_run(
    cfg: ArenaRunCfg,
    output_dir: str | Path,
    video_cfg: VideoRecordingCfg | None = None,
) -> ArenaRunResult:
    """Build and execute one typed Arena run, then return its result."""
    metrics_per_rebuild: list[MetricsDataCollection] = []
    output_dir = str(output_dir)
    video_cfg = video_cfg or VideoRecordingCfg(video_base_dir=output_dir)
    episodes_per_rebuild = _split_episodes_across_rebuilds(
        cfg.rollout_limit.num_episodes,
        cfg.num_rebuilds,
        cfg.name,
    )

    for rebuild_index, num_episodes in enumerate(episodes_per_rebuild):
        env = None
        policy = None
        try:
            rebuild_video_cfg = replace(
                video_cfg,
                video_base_dir=output_dir,
                camera_name_prefix=f"robot-cam-rebuild{rebuild_index}",
            )
            env = _build_environment_from_cfg(cfg, rebuild_video_cfg.render_mode)
            results_path = os.path.join(output_dir, f"episode_results_rebuild{rebuild_index}.jsonl")
            env.unwrapped.episode_recorder.set_job_name(cfg.name)
            env.unwrapped.episode_recorder.set_output_path(results_path)

            policy = _build_policy_from_cfg(cfg)
            num_steps, num_episodes = _resolve_rollout_limit(
                cfg,
                policy,
                num_episodes,
            )
            env = wrap_env_for_video(env, rebuild_video_cfg, num_steps, num_episodes)
            metrics = rollout_policy(env, policy, num_steps=num_steps, num_episodes=num_episodes)
            if metrics is not None:
                metrics_per_rebuild.append(metrics)
        finally:
            close_run_resources(policy, env)

    return ArenaRunResult(
        run_name=cfg.name,
        status=RunStatus.COMPLETED,
        metrics=aggregate_metrics(metrics_per_rebuild) if metrics_per_rebuild else None,
    )


def _build_environment_from_cfg(
    cfg: ArenaRunCfg,
    render_mode: str | None,
) -> gym.Env:
    """Compile and instantiate a run's environment."""
    arena_builder = build_arena_builder_from_run_cfg(cfg)
    _, env_cfg, env_kwargs = arena_builder.build_registered()
    if env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{cfg.name}"
    return arena_builder.make_registered(env_cfg, env_kwargs, render_mode=render_mode)


def build_arena_builder_from_run_cfg(cfg: ArenaRunCfg) -> ArenaEnvBuilder:
    """Build an Arena environment builder from one typed run config."""
    hydra_overrides = overrides_from_dict(cfg.variations)
    # TODO(cvolk, 2026-07-07): [typed-config-migration] Remove the legacy branch when graph environments
    # have typed configs and no longer require the argparse construction path.
    return (
        build_arena_builder_from_legacy_graph(
            cfg.environment,
            device=cfg.environment_builder.device,
            language_instruction=cfg.environment_builder.language_instruction,
            hydra_overrides=hydra_overrides,
        )
        if isinstance(cfg.environment, LegacyGraphEnvironmentCfg)
        else _build_arena_builder_from_cfg(cfg, hydra_overrides)
    )


def _build_arena_builder_from_cfg(cfg: ArenaRunCfg, hydra_overrides: list[str]) -> ArenaEnvBuilder:
    """Build an Arena environment builder from a registered typed config."""
    # ArenaEnvBuilder imports pxr modules that must not load before SimulationApp.
    # Keep this runtime import deferred even though the type-only import is at the top.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    environment_factory_type = EnvironmentRegistry().get_factory_type_for_cfg(cfg.environment)
    arena_environment = environment_factory_type().build(cfg.environment)
    return ArenaEnvBuilder(
        arena_environment,
        cfg.environment_builder,
        hydra_overrides=hydra_overrides,
    )


def _build_policy_from_cfg(cfg: ArenaRunCfg) -> PolicyBase:
    """Instantiate the registered policy configured by a run."""
    policy_cfg = _policy_cfg_for_num_envs(cfg.policy, cfg.environment_builder.num_envs)
    policy_type = PolicyRegistry().get_policy_type_for_cfg(policy_cfg)
    return policy_type(policy_cfg)


def _policy_cfg_for_num_envs(policy_cfg: PolicyCfg, num_envs: int) -> PolicyCfg:
    """Align legacy policy batch-size fields with the environment batch size."""
    # TODO(cvolk, 2026-07-06): [typed-config-migration] Remove the duplicated policy ``num_envs`` fields and
    # this adapter once policies receive the environment batch size as runtime context.
    if "num_envs" not in {config_field.name for config_field in fields(policy_cfg)}:
        return policy_cfg
    return replace(policy_cfg, num_envs=num_envs)


def _resolve_rollout_limit(
    cfg: ArenaRunCfg,
    policy: PolicyBase,
    num_episodes: int | None,
) -> tuple[int | None, int | None]:
    """Resolve a configured rollout limit or use the policy's intrinsic length."""
    num_steps = cfg.rollout_limit.num_steps
    if num_steps is None and num_episodes is None:
        assert (
            policy.has_length()
        ), f"Run '{cfg.name}' must configure num_steps or num_episodes because its policy has no intrinsic length"
        num_steps = policy.length()
        assert num_steps is not None and num_steps > 0, f"Policy for run '{cfg.name}' has no usable length"
    return num_steps, num_episodes


def _split_episodes_across_rebuilds(
    num_episodes: int | None,
    num_rebuilds: int,
    run_name: str,
) -> list[int | None]:
    """Split a run's episode budget as evenly as possible across rebuilds."""
    if num_episodes is None:
        return [None] * num_rebuilds
    assert num_episodes >= num_rebuilds, (
        f"Run '{run_name}': num_episodes ({num_episodes}) must be >= num_rebuilds "
        f"({num_rebuilds}) so each rebuild runs at least one episode"
    )
    base, remainder = divmod(num_episodes, num_rebuilds)
    return [base + int(rebuild_index < remainder) for rebuild_index in range(num_rebuilds)]
