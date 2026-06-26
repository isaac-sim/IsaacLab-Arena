# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import gc
import json
import os
import torch
import traceback
from gymnasium.wrappers import RecordVideo
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.job_manager import Job, JobManager, Status
from isaaclab_arena.evaluation.policy_runner import get_policy_cls, prepare_env_cfg_for_datagen, rollout_policy
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app
from isaaclab_arena.utils.reload_modules import reload_arena_modules
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase


def load_env(
    arena_env_args: list[str], job_name: str, render_mode: str | None = None, disable_auto_reset: bool = False
):

    reload_arena_modules()

    args_parser = get_isaaclab_arena_environments_cli_parser()

    arena_env_args_cli = args_parser.parse_args(arena_env_args)
    arena_builder = get_arena_builder_from_cli(arena_env_args_cli)

    env_name, env_cfg = arena_builder.build_registered()

    # Set unique dataset filename for this job to avoid file locking conflicts
    if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{job_name}"

    # Datagen: disable the env's in-step() auto-reset (and metric recorders) so the rollout
    # loop can drive episode boundaries and resets explicitly (see prepare_env_cfg_for_datagen).
    reset_terms = prepare_env_cfg_for_datagen(env_cfg) if disable_auto_reset else None

    env = arena_builder.make_registered(env_cfg, render_mode=render_mode)
    # Don't reset here - rollout_policy() will reset the env. Every reset triggers a new episode, initializing recorder & creating a new hdf5 entry.
    return env, reset_terms


def enable_cameras_if_required(eval_jobs_config: dict, args_cli: argparse.Namespace) -> None:
    """
    Check if any job requires cameras and enable them in args_cli if needed. Users can set
    enable_cameras: true in individual job config, or add --enable_cameras to the CLI.
    Camera support must be enabled when the simulation starts, not during individual job execution.

    Args:
        eval_jobs_config: Dictionary containing job configurations
        args_cli: CLI arguments namespace to modify
    """
    for job_dict in eval_jobs_config["jobs"]:
        if "arena_env_args" in job_dict and job_dict["arena_env_args"].get("enable_cameras", False):
            if not hasattr(args_cli, "enable_cameras") or not args_cli.enable_cameras:
                args_cli.enable_cameras = True
            break


def get_policy_from_job(job: Job) -> "PolicyBase":
    """
    Create a policy from a job configuration. Two paths are supported:
    1. JSON → dict → ConfigDataclass → init cls (preferred, if policy has config_class)
    2. JSON → dict → CLI args → init cls (if policy has add_args_to_parser() and from_args())
    """
    # Each job can be evaluated with a different policy checkpoint, or even a different policy type
    policy_cls = get_policy_cls(job.policy_type)

    policy_config_dict = dict(job.policy_config_dict)
    # Align policy num_envs with env when the policy config supports it (optional key)
    if hasattr(policy_cls, "config_class") and policy_cls.config_class is not None:
        config_fields = {f.name for f in dataclasses.fields(policy_cls.config_class)}
        if "num_envs" in config_fields:
            policy_config_dict["num_envs"] = job.num_envs

    # Use direct from_dict if the policy class has config_class defined
    if hasattr(policy_cls, "config_class") and policy_cls.config_class is not None:
        # Use the inherited from_dict() method from PolicyBase
        policy = policy_cls.from_dict(policy_config_dict)
    else:
        policy_args_parser = get_isaaclab_arena_cli_parser()
        policy_added_args_parser = policy_cls.add_args_to_parser(policy_args_parser)
        policy_args = policy_added_args_parser.parse_args(policy_config_dict)
        policy = policy_cls.from_args(policy_args)
    return policy


def build_datagen_collector(job: Job, datagen_defaults: dict | None, env):
    """Build a per-job datagen collector, or ``None`` if datagen is not configured.

    The effective config is the top-level ``datagen`` defaults overridden by the
    job's own ``datagen`` block. Per-episode HDF5 files are written under
    ``{output_dir}/{job.name}/episode_NNNN/dataset.h5``. A camera is taken from
    ``camera_position`` (+ optional ``camera_target``, default origin); otherwise
    the env's default view is used. Lazily imports the datagen package so core
    stays decoupled unless collection is requested.
    """
    merged = {**(datagen_defaults or {}), **(job.datagen or {})}
    if not merged.get("output_dir"):
        return None

    from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory
    from isaaclab_arena_datagen.collection.collector import DatagenCollector, DatagenCollectorConfig
    from isaaclab_arena_datagen.utils.constants import DEFAULT_ROTATION_EPS_RAD, DEFAULT_TRANSLATION_EPS_M

    cameras = None
    if merged.get("camera_position") is not None:
        target = tuple(merged["camera_target"]) if merged.get("camera_target") is not None else (0.0, 0.0, 0.0)
        cameras = [
            CameraViewTrajectory(
                position=tuple(merged["camera_position"]),
                target=target,
                focal_length_mm=merged.get("focal_length_mm", 24.0),
            )
        ]

    cfg = DatagenCollectorConfig(
        output_dir=os.path.join(merged["output_dir"], job.name),
        cameras=cameras,
        width=merged.get("width", 640),
        height=merged.get("height", 480),
        mesh_sample_spacing=merged.get("mesh_sample_spacing", 0.01),
        dynamic_translation_eps=merged.get("dynamic_translation_eps", DEFAULT_TRANSLATION_EPS_M),
        dynamic_rotation_eps=merged.get("dynamic_rotation_eps", DEFAULT_ROTATION_EPS_RAD),
    )
    print(f"[INFO] Datagen collection enabled for job '{job.name}' -> {cfg.output_dir}/episode_NNNN/dataset.h5")
    return DatagenCollector.from_env(env, cfg, env_name=None)


def _capture_sim_info(env):
    """Snapshot per-job sim/render settings from a live env into a manifest.SimInfo."""
    from isaaclab_arena_datagen.manifest import SimInfo

    cfg = env.unwrapped.cfg
    sim = getattr(cfg, "sim", None)
    render = getattr(sim, "render", None) if sim is not None else None
    return SimInfo(
        dt=getattr(sim, "dt", None),
        render_interval=getattr(sim, "render_interval", None),
        decimation=getattr(cfg, "decimation", None),
        episode_length_s=getattr(cfg, "episode_length_s", None),
        render_carb_settings=dict(getattr(render, "carb_settings", {}) or {}),
    )


def _collect_garbage_and_clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _close_policy(policy: "PolicyBase | None") -> None:
    try:
        if policy is not None:
            policy.close()
    finally:
        _collect_garbage_and_clear_cuda_cache()


def _close_env(env) -> None:
    if env is None:
        return
    try:
        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
    finally:
        try:
            # cleanup managers, including recorder manager closing hdf5 file
            env.close()
        finally:
            _collect_garbage_and_clear_cuda_cache()


def _close_job_resources(policy: "PolicyBase | None", env) -> None:
    try:
        _close_policy(policy)
    finally:
        _close_env(env)


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, unknown = args_parser.parse_known_args()

    # Load job configuration before starting simulation to check requirements
    add_eval_runner_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()
    assert not args_cli.distributed, "Distributed evaluation is not supported yet"

    assert os.path.exists(
        args_cli.eval_jobs_config
    ), f"eval_jobs_config file does not exist: {args_cli.eval_jobs_config}"

    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        eval_jobs_config = json.load(f)

    # Optional top-level datagen defaults, applied to every job (a per-job "datagen"
    # block overrides these). When present, datagen data collection runs alongside
    # each rollout, writing one HDF5 file per episode.
    datagen_defaults = eval_jobs_config.get("datagen")

    manifest_root = (datagen_defaults or {}).get("output_dir")
    manifest_jobs = []  # list[manifest.JobRecord], populated as jobs finish
    manifest_description = args_cli.datagen_description or (datagen_defaults or {}).get("description")
    if manifest_root:
        from isaaclab_arena_datagen import manifest as _manifest

        manifest_git = _manifest.capture_git_info()
        manifest_system = _manifest.capture_system_info(args_cli.device)

    # Check if any job requires cameras and enable them if needed before starting simulation
    enable_cameras_if_required(eval_jobs_config, args_cli)
    # Datagen collection renders dedicated cameras, which requires camera support.
    if datagen_defaults or any(job_dict.get("datagen") for job_dict in eval_jobs_config["jobs"]):
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli):
        job_manager = JobManager(eval_jobs_config["jobs"])
        metrics_logger = MetricsLogger()

        job_manager.print_jobs_info()

        if args_cli.video:
            os.makedirs(args_cli.video_dir, exist_ok=True)
            print(f"[INFO] Video recording enabled. Videos will be saved to: {args_cli.video_dir}")

        for job in job_manager:
            if job is not None:
                env = None
                policy = None
                collector = None
                job_sim_info = None
                try:
                    render_mode = "rgb_array" if args_cli.video else None
                    # Datagen is active for this job when an output_dir resolves (top-level
                    # defaults overridden by the job's own datagen block) - mirror build_datagen_collector.
                    job_datagen = {**(datagen_defaults or {}), **(job.datagen or {})}
                    is_datagen = bool(job_datagen.get("output_dir"))
                    env, datagen_reset_terms = load_env(
                        job.arena_env_args, job.name, render_mode=render_mode, disable_auto_reset=is_datagen
                    )

                    policy = get_policy_from_job(job)

                    # Resolve simulation length: num_steps and num_episodes are mutually exclusive.
                    # Priority: job config -> policy length -> CLI default
                    if job.num_steps is None and job.num_episodes is None:
                        if policy.has_length():
                            job.num_steps = policy.length()
                        else:
                            job.num_steps = args_cli.num_steps

                    if args_cli.video:
                        if job.num_steps is not None:
                            video_length = job.num_steps
                        else:
                            video_length = job.num_episodes * env.unwrapped.max_episode_length
                        video_kwargs = {
                            "video_folder": os.path.join(args_cli.video_dir, job.name),
                            "step_trigger": lambda step: step == 0,
                            "video_length": video_length,
                            "disable_logger": True,
                        }
                        print(f"[INFO] Recording video for job '{job.name}' -> {video_kwargs['video_folder']}")
                        env = RecordVideo(env, **video_kwargs)

                    collector = build_datagen_collector(job, datagen_defaults, env)
                    job_sim_info = _capture_sim_info(env) if collector is not None else None

                    metrics = rollout_policy(
                        env,
                        policy,
                        num_steps=job.num_steps,
                        num_episodes=job.num_episodes,
                        language_instruction=job.language_instruction,
                        collector=collector,
                        datagen_reset_terms=datagen_reset_terms,
                        max_episode_length=int(env.unwrapped.max_episode_length) if is_datagen else None,
                    )

                    job_manager.complete_job(job, metrics=metrics, status=Status.COMPLETED)

                    # users may not specify metrics for a task, although it's not recommended
                    if metrics is not None:
                        metrics_logger.append_job_metrics(job.name, metrics)

                except Exception as e:
                    job_manager.complete_job(job, metrics={}, status=Status.FAILED)
                    print(f"Job {job.name} failed with error: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    if not args_cli.continue_on_error:
                        raise

                finally:
                    try:
                        # Release datagen cameras BEFORE tearing down the stage, so
                        # their replicator annotators do not leak into the next job.
                        if collector is not None:
                            collector.close(env)
                            try:
                                manifest_jobs.append(
                                    _manifest.build_job_record(
                                        name=job.name,
                                        status=job.status.value,
                                        policy_type=job.policy_type,
                                        policy_config=job.policy_config_dict,
                                        language_instruction=job.language_instruction,
                                        arena_env_args=job.arena_env_args,
                                        datagen_settings=_manifest.clean_datagen_settings(
                                            dataclasses.asdict(collector.config)
                                        ),
                                        sim=job_sim_info or _manifest.SimInfo(),
                                        sequence_dicts=list(collector.sequences),
                                        root=manifest_root,
                                    )
                                )
                            except Exception as exc:  # never let manifest bookkeeping fail a run
                                print(f"[datagen] Warning: failed to record job '{job.name}' in manifest: {exc}")
                    finally:
                        try:
                            _close_job_resources(policy, env)
                        finally:
                            policy = None
                            env = None
                            collector = None
                            _collect_garbage_and_clear_cuda_cache()

        if manifest_root and manifest_jobs:
            import datetime

            created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            manifest = _manifest.build_manifest(
                created_at=created_at,
                description=manifest_description,
                generator_tool="isaaclab_arena.evaluation.eval_runner",
                git=manifest_git,
                system=manifest_system,
                input_config=eval_jobs_config,
                jobs=manifest_jobs,
            )
            out_path = os.path.join(manifest_root, "manifest.json")
            if _manifest.write_manifest(out_path, manifest):
                print(f"[datagen] Wrote dataset manifest -> {out_path}")

        job_manager.print_jobs_info()
        metrics_logger.print_metrics()


if __name__ == "__main__":
    main()
