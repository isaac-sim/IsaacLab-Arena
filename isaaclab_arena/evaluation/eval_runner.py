# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import torch
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.job_manager import Job, JobManager, Status
from isaaclab_arena.evaluation.policy_runner import get_policy_cls, rollout_policy
from isaaclab_arena.metrics.aggregate_metrics import aggregate_metrics
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir, wrap_env_for_video
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase


def load_env(
    arena_env_args: list[str],
    job_name: str,
    variations: list[str] | None = None,
    render_mode: str | None = None,
):

    args_parser = get_isaaclab_arena_environments_cli_parser()

    arena_env_args_cli = args_parser.parse_args(arena_env_args)
    arena_builder = get_arena_builder_from_cli(arena_env_args_cli, hydra_overrides=variations)

    env_name, env_cfg, env_kwargs = arena_builder.build_registered()

    # Set unique dataset filename for this job to avoid file locking conflicts
    if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{job_name}"

    env = arena_builder.make_registered(env_cfg, env_kwargs, render_mode=render_mode)
    # Don't reset here - rollout_policy() will reset the env. Every reset triggers a new episode, initializing recorder & creating a new hdf5 entry.
    return env


def list_variations(eval_jobs_config: dict) -> None:
    """Print the Hydra-configurable variations for each job's environment."""
    job_manager = JobManager(eval_jobs_config["jobs"])
    for job in job_manager.all_jobs:
        args_parser = get_isaaclab_arena_environments_cli_parser()
        arena_env_args_cli = args_parser.parse_args(job.arena_env_args)
        arena_builder = get_arena_builder_from_cli(arena_env_args_cli, hydra_overrides=job.variations)
        print(f"=== Variations for job '{job.name}' ===", flush=True)
        print(arena_builder.get_variations_catalogue_as_string(), flush=True)


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


def _split_episodes_across_rebuilds(num_episodes: int | None, num_rebuilds: int, job_name: str) -> list[int | None]:
    """Split a job's total ``num_episodes`` as evenly as possible across its rebuilds.

    ``num_episodes`` is the total accumulated across rebuilds. The first ``remainder`` rebuilds
    get one extra episode when the split is uneven (e.g. ``num_episodes=5, num_rebuilds=2`` ->
    ``[3, 2]``). Returns a list of ``None`` (one per rebuild) when the job is length-driven by
    steps rather than episodes.
    """
    if num_episodes is None:
        return [None] * num_rebuilds
    assert num_episodes >= num_rebuilds, (
        f"Job '{job_name}': num_episodes ({num_episodes}) must be >= num_rebuilds"
        f" ({num_rebuilds}) so each rebuild runs at least one episode"
    )
    # Give every rebuild ``base`` episodes, then hand out the leftover episodes one at a
    # time to the first ``remainder`` rebuilds.
    base, remainder = divmod(num_episodes, num_rebuilds)
    episodes_per_rebuild = [base] * num_rebuilds
    for rebuild_idx in range(remainder):
        episodes_per_rebuild[rebuild_idx] += 1
    return episodes_per_rebuild


def _run_chunk(chunk_label: str, chunk_jobs: list[dict]) -> int:
    """Run ``chunk_jobs`` in a fresh ``eval_runner`` subprocess and return its exit code."""
    print(f"[eval_runner] {chunk_label}", flush=True)
    # Serialize this chunk's jobs to a temp config the child loads via --eval_jobs_config.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"jobs": chunk_jobs}, tmp)
        chunk_path = Path(tmp.name)
    # Re-run this invocation in the child, with --eval_jobs_config appended so it wins over
    # the master config (argparse keeps the last value).
    this_invocation = sys.argv
    config_override = ["--eval_jobs_config", str(chunk_path)]
    child_cmd = [sys.executable, *this_invocation, *config_override]
    try:
        result = subprocess.run(child_cmd, check=False)
    finally:
        # Remove the temp chunk config now that the child has loaded it.
        chunk_path.unlink(missing_ok=True)
    return result.returncode


def _run_in_chunks(args_cli: argparse.Namespace, master_cfg: dict) -> None:
    """Run each chunk of ``master_cfg['jobs']`` in a fresh ``eval_runner`` subprocess."""
    jobs = master_cfg["jobs"]
    chunk_size = args_cli.chunk_size
    if chunk_size <= 0:
        raise ValueError(f"--chunk_size must be positive, got {chunk_size}")
    n_chunks = math.ceil(len(jobs) / chunk_size)
    print(f"[eval_runner] {len(jobs)} jobs → {n_chunks} chunks of <= {chunk_size}", flush=True)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(jobs))
        chunk_label = f"chunk {chunk_idx + 1}/{n_chunks}: jobs {start}..{end - 1}"
        returncode = _run_chunk(chunk_label, jobs[start:end])
        if returncode != 0:
            print(f"[eval_runner] chunk {chunk_idx} failed (exit {returncode}).", flush=True)
            sys.exit(returncode)


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

    # Print the variations catalogue for each job's environment and exit.
    if args_cli.list_variations:
        with SimulationAppContext(args_cli):
            list_variations(eval_jobs_config)
        return

    # Chunked dispatch (--chunk_size N). Splits this config across subprocesses so each
    # gets a fresh SimulationApp. Required for long sweeps because some host memory leaks
    # each cycle and is only reclaimed when the process exits — in-process teardown can't
    # release it.
    if args_cli.chunk_size is not None and len(eval_jobs_config["jobs"]) > args_cli.chunk_size:
        # TODO(cvolk): aggregate per-chunk metrics into one centralized view. Each chunk
        # subprocess currently prints its own MetricsLogger summary and nothing is merged
        # or persisted (save_metrics_to_file() is unused). Follow-up: have each chunk write
        # metrics JSON to a temp file (forward --metrics_file), then merge + print/save here.
        _run_in_chunks(args_cli, eval_jobs_config)
        return

    # Check if any job requires cameras and enable them if needed before starting simulation
    if args_cli.record_camera_video:
        args_cli.enable_cameras = True
    enable_cameras_if_required(eval_jobs_config, args_cli)

    with SimulationAppContext(args_cli):
        job_manager = JobManager(eval_jobs_config["jobs"])
        metrics_logger = MetricsLogger()

        job_manager.print_jobs_info()

        # One reverse-dated run directory shared by all jobs; each job gets a subdirectory within it.
        recording = args_cli.record_viewport_video or args_cli.record_camera_video
        run_video_dir = timestamped_run_dir(args_cli.video_base_dir) if recording else args_cli.video_base_dir

        if args_cli.record_viewport_video:
            os.makedirs(run_video_dir, exist_ok=True)
            print(f"[INFO] Video recording enabled. Videos will be saved to: {run_video_dir}")

        for job in job_manager:
            if job is None:
                continue
            env = None
            policy = None

            metrics_per_run: list[MetricsDataCollection] = []

            # num_episodes is the total across rebuilds, so split it over the rebuilds.
            num_episodes_per_rebuild = _split_episodes_across_rebuilds(job.num_episodes, job.num_rebuilds, job.name)

            # Rebuild the environment and re-run the rollout job.num_rebuilds times, then
            # aggregate the metrics across rebuilds into a single result.
            for rebuild_idx in range(job.num_rebuilds):
                try:
                    # Per-job video output directory; cameras are tagged with the rebuild index.
                    video_cfg = VideoRecordingCfg(
                        record_viewport_video=args_cli.record_viewport_video,
                        record_camera_video=args_cli.record_camera_video,
                        video_base_dir=os.path.join(run_video_dir, job.name),
                        camera_name_prefix=f"robot-cam-rebuild{rebuild_idx}",
                    )
                    env = load_env(
                        job.arena_env_args, job.name, variations=job.variations, render_mode=video_cfg.render_mode
                    )

                    policy = get_policy_from_job(job)

                    # Episodes allotted to this rebuild (None when the job is length-driven by steps).
                    num_episodes_this_rebuild = num_episodes_per_rebuild[rebuild_idx]

                    # Resolve simulation length: num_steps and num_episodes are mutually exclusive.
                    # Priority: job config -> policy length -> CLI default
                    if job.num_steps is None and num_episodes_this_rebuild is None:
                        if policy.has_length():
                            job.num_steps = policy.length()
                        else:
                            job.num_steps = args_cli.num_steps

                    env = wrap_env_for_video(env, video_cfg, job.num_steps, num_episodes_this_rebuild)

                    metrics = rollout_policy(
                        env,
                        policy,
                        num_steps=job.num_steps,
                        num_episodes=num_episodes_this_rebuild,
                        language_instruction=job.language_instruction,
                    )

                    job_manager.complete_job(job, metrics=metrics, status=Status.COMPLETED)

                    # users may not specify metrics for a task, although it's not recommended
                    if metrics is not None:
                        metrics_per_run.append(metrics)

                except Exception as e:
                    job_manager.complete_job(job, metrics={}, status=Status.FAILED)
                    print(f"Job {job.name} failed with error: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    if not args_cli.continue_on_error:
                        raise

                finally:
                    try:
                        _close_job_resources(policy, env)
                    finally:
                        policy = None
                        env = None
                        _collect_garbage_and_clear_cuda_cache()

            # Aggregate the metrics from the different experiments into a single view.
            if metrics_per_run:
                aggregated_metrics = aggregate_metrics(metrics_per_run)
                metrics_logger.append_job_metrics(job.name, aggregated_metrics)

        job_manager.print_jobs_info()
        metrics_logger.print_metrics()


if __name__ == "__main__":
    main()
