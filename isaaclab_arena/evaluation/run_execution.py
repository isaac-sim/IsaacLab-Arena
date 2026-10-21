# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and execute typed Arena runs and experiments for an evaluation frontend."""

from __future__ import annotations

import os
import traceback
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RunStatus
from isaaclab_arena.evaluation.condition_episode_budget import CONDITION_EPISODE_BUDGET_FILENAME
from isaaclab_arena.evaluation.experiment_timings import write_run_timings
from isaaclab_arena.evaluation.legacy_graph_environment_cli import (
    LegacyGraphEnvironmentCfg,
    build_arena_builder_from_legacy_graph,
)
from isaaclab_arena.evaluation.policy_runner import rollout_policy
from isaaclab_arena.evaluation.resource_cleanup import close_run_resources
from isaaclab_arena.metrics.aggregate_metrics import aggregate_metrics
from isaaclab_arena.recording.episode_results import read_episode_records
from isaaclab_arena.utils.timer import print_timer_stats, reset_timer_stats
from isaaclab_arena.variations.variations_hydra import overrides_from_dict
from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


def execute_experiment(
    experiment_cfg: ArenaExperimentCfg,
    output_dir: Path,
    record_viewport_video: bool = False,
    record_camera_video: bool = False,
    continue_on_error: bool = False,
) -> list[ArenaRunResult]:
    """Execute an experiment's runs in order and return their results.

    Args:
        experiment_cfg: Named Run configurations that make up the Experiment.
        output_dir: Directory containing one output subdirectory per run.
        record_viewport_video: Whether to record the viewport for each run.
        record_camera_video: Whether to record observation cameras for each run.
        continue_on_error: Whether to continue with later runs after one fails.

    Returns:
        One result per attempted run, in execution order.
    """
    results = []
    for run_cfg in experiment_cfg.runs.values():
        print(f"Running run '{run_cfg.name}'", flush=True)
        run_output_dir = output_dir / run_cfg.name
        # Clear timers.
        reset_timer_stats()
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

        print_timer_stats()
        write_run_timings(run_output_dir)
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
    episodes_per_rebuild = _episodes_per_rebuild(cfg)
    condition_snapshot = None

    for rebuild_index, num_episodes in enumerate(episodes_per_rebuild):
        env = None
        policy = None
        condition_budget = None
        rollout_error = None
        try:
            rebuild_video_cfg = replace(
                video_cfg,
                video_base_dir=output_dir,
                camera_name_prefix=f"robot-cam-rebuild{rebuild_index}",
            )
            rebuild_cfg = _cfg_for_rebuild(cfg, rebuild_index)
            env = _build_environment_from_cfg(
                rebuild_cfg,
                rebuild_video_cfg.render_mode,
                output_dir=output_dir,
                rebuild_index=rebuild_index,
            )
            results_path = os.path.join(output_dir, f"episode_results_rebuild{rebuild_index}.jsonl")
            env.unwrapped.episode_recorder.set_job_name(cfg.name)
            env.unwrapped.episode_recorder.set_output_path(results_path)
            if cfg.rollout_limit.object_set_episodes is not None:
                condition_budget = env.unwrapped.condition_episode_budget
                condition_budget.set_output_path(Path(output_dir) / CONDITION_EPISODE_BUDGET_FILENAME)

            policy = _build_policy_from_cfg(rebuild_cfg)
            num_steps, num_episodes = _resolve_rollout_limit(
                rebuild_cfg,
                policy,
                num_episodes,
            )
            env = wrap_env_for_video(env, rebuild_video_cfg, num_steps, num_episodes)
            rollout_options = {}
            if cfg.rollout_limit.object_set_episodes is not None:
                rollout_options["max_steps"] = cfg.rollout_limit.object_set_episodes.max_steps
            metrics = rollout_policy(env, policy, num_steps=num_steps, num_episodes=num_episodes, **rollout_options)
            if metrics is not None:
                metrics_per_rebuild.append(metrics)
        except BaseException as error:
            rollout_error = error
            raise
        finally:
            try:
                close_run_resources(policy, env)
            except BaseException as error:
                rollout_error = error
                raise
            finally:
                if condition_budget is not None:
                    completed = rollout_error is None and condition_budget.is_complete
                    reason = (
                        type(rollout_error).__name__
                        if rollout_error is not None
                        else ("episode_budget_met" if completed else "step_limit")
                    )
                    condition_snapshot = condition_budget.write(
                        status="completed" if completed else "incomplete", reason=reason
                    )

    return ArenaRunResult(
        run_name=cfg.name,
        status=(
            RunStatus.INCOMPLETE
            if condition_snapshot is not None and not condition_snapshot["sampling_complete"]
            else RunStatus.COMPLETED
        ),
        metrics=aggregate_metrics(metrics_per_rebuild) if metrics_per_rebuild else None,
        condition_episode_budget=condition_snapshot,
    )


def _cfg_for_rebuild(cfg: ArenaRunCfg, rebuild_index: int) -> ArenaRunCfg:
    """Return an isolated config with this rebuild's seed and replay source."""
    cfg = deepcopy(cfg)
    cfg.environment_builder.seed += rebuild_index
    if cfg.recorded_variation_samples_paths is not None:
        cfg.environment_builder.recorded_variation_samples_path = cfg.recorded_variation_samples_paths[rebuild_index]
        cfg.recorded_variation_samples_paths = None
    return cfg


def _build_environment_from_cfg(
    cfg: ArenaRunCfg,
    render_mode: str | None,
    output_dir: str | Path,
    rebuild_index: int = 0,
) -> gym.Env:
    """Compile and instantiate a run's environment.

    Args:
        cfg: The run whose environment is being built.
        render_mode: Render mode required by the enabled video recorders.
        output_dir: Directory the run's recorded dataset is exported to.
        rebuild_index: Index of the rebuild this environment serves.
    """
    # Copy runtime-derived recorder output options into the builder cfg so
    # ArenaEnvBuilder.build_env_cfg sees them without mutating the caller's run config.
    run_cfg = deepcopy(cfg)
    run_cfg.environment_builder.recorder_dataset_export_dir_path = str(output_dir)
    run_cfg.environment_builder.recorder_dataset_filename = f"dataset_{cfg.name}_rebuild{rebuild_index}"
    arena_builder = build_arena_builder_from_run_cfg(run_cfg)
    condition_cfg = cfg.rollout_limit.object_set_episodes
    if condition_cfg is not None:
        from isaaclab.sim import MultiAssetSpawnerCfg

        from isaaclab_arena.assets.object_set import RigidObjectSet

        object_sets = [
            asset for asset in arena_builder.arena_env.scene.assets.values() if isinstance(asset, RigidObjectSet)
        ]
        assert (
            len(object_sets) == 1 and object_sets[0].name == condition_cfg.object_set
        ), "Condition budgets require the scene's single RigidObjectSet to match object_set"
        object_set = object_sets[0]
        for asset in arena_builder.arena_env.scene.assets.values():
            spawn = getattr(getattr(asset, "object_cfg", None), "spawn", None)
            assert asset is object_set or not isinstance(
                spawn, MultiAssetSpawnerCfg
            ), "Condition budgets require the object set to be the scene's only multi-asset spawner"
        assert not object_set.random_choice, "Condition budgets require an ordered RigidObjectSet"
        names = [member.name for member in object_set.objects]
        assert len(set(names)) == len(names), "Object-set member names must be unique"
        assert set(names) == set(condition_cfg.episodes_per_member), "Specify a quota for every object-set member"
        assert cfg.environment_builder.num_envs >= len(names), "Every condition needs at least one environment"
        object_set.assign_variants(cfg.environment_builder.num_envs)
        members_by_env = [names[index] for index in object_set.variant_indices_by_env]
        assigned_spawn = deepcopy(object_set.object_cfg.spawn.to_dict())
        env_cfg, env_kwargs = arena_builder.build_env_cfg()
        assert names == [
            member.name for member in object_set.objects
        ], "Environment construction changed the object-set members"
        assert members_by_env == [
            names[index] for index in object_set.variant_indices_by_env
        ], "Environment construction changed the condition assignment"
        scene_name, _ = object_set.get_object_cfg()
        resolved_spawn = getattr(env_cfg.scene, scene_name).spawn
        assert (
            resolved_spawn.to_dict() == assigned_spawn
        ), "The final scene configuration must preserve the object-set spawn assignment"
        for name, asset_cfg in vars(env_cfg.scene).items():
            assert name == scene_name or not isinstance(
                getattr(asset_cfg, "spawn", None), MultiAssetSpawnerCfg
            ), "Condition budgets require the object set to be the final scene's only multi-asset spawner"
        resolved_entries = (
            resolved_spawn.assets_cfg if isinstance(resolved_spawn, MultiAssetSpawnerCfg) else [resolved_spawn]
        )
        member_definitions = {}
        for index, member in enumerate(object_set.objects):
            slot_configs = (
                [
                    resolved_entries[env_id].to_dict()
                    for env_id, name in enumerate(members_by_env)
                    if name == member.name
                ]
                if len(names) > 1
                else [resolved_entries[0].to_dict()]
            )
            assert all(
                spawn == slot_configs[0] for spawn in slot_configs
            ), "Slots sharing a member must use the same resolved spawn configuration"
            member_definitions[member.name] = {
                "index": index,
                "declared_spawn": member.object_cfg.spawn.to_dict(),
                "resolved_spawn": slot_configs[0],
            }
        env_kwargs["condition_episode_budget_cfg"] = condition_cfg
        env_kwargs["condition_members_by_env"] = members_by_env
        env_kwargs["condition_member_definitions"] = member_definitions
        return arena_builder.make_registered(env_cfg, env_kwargs, render_mode=render_mode)
    return arena_builder.make_registered(render_mode=render_mode)


def build_arena_builder_from_run_cfg(cfg: ArenaRunCfg) -> ArenaEnvBuilder:
    """Build an Arena environment builder from one typed run config."""
    hydra_overrides = overrides_from_dict(cfg.variations)
    # TODO(cvolk, 2026-07-07): [typed-config-migration] Remove the legacy branch when graph environments
    # have typed configs and no longer require the argparse construction path.
    return (
        build_arena_builder_from_legacy_graph(
            cfg.environment,
            environment_builder=cfg.environment_builder,
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
    if (
        cfg.environment_builder.recorded_variation_samples_path is not None
        or cfg.rollout_limit.object_set_episodes is not None
    ):
        assert not policy.has_length(), (
            f"Run '{cfg.name}' cannot use recorded replay or condition budgets with a finite-action policy; "
            "finite policies do not restart their action sequence across episodes"
        )
    num_steps = cfg.rollout_limit.num_steps
    if num_steps is None and num_episodes is None:
        assert (
            policy.has_length()
        ), f"Run '{cfg.name}' must configure num_steps or num_episodes because its policy has no intrinsic length"
        num_steps = policy.length()
        assert num_steps is not None and num_steps > 0, f"Policy for run '{cfg.name}' has no usable length"
    return num_steps, num_episodes


def _episodes_per_rebuild(cfg: ArenaRunCfg) -> list[int | None]:
    """Resolve one rollout episode budget per environment rebuild."""
    if cfg.rollout_limit.object_set_episodes is not None:
        return [sum(cfg.rollout_limit.object_set_episodes.episodes_per_member.values())]
    if cfg.recorded_variation_samples_paths is not None:
        if cfg.rollout_limit.num_episodes is None:
            return [_recorded_episode_count(path) for path in cfg.recorded_variation_samples_paths]

    total_num_episodes = cfg.rollout_limit.num_episodes
    replay_path = cfg.environment_builder.recorded_variation_samples_path
    if replay_path is not None and total_num_episodes is None:
        return [_recorded_episode_count(replay_path)] * cfg.num_rebuilds
    return _split_episodes_across_rebuilds(total_num_episodes, cfg.num_rebuilds, cfg.name)


def _recorded_episode_count(path: str) -> int:
    """Return the number of episode rows in one replay JSONL."""
    records = read_episode_records(path)
    assert records, f"Recorded variation samples must list at least one episode: {path}"
    return len(records)


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
