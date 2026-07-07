# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.arena_experiment import ExperimentStatus
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.experiment_execution import build_and_run_experiment
from isaaclab_arena.evaluation.job_manager import JobManager, Status
from isaaclab_arena.evaluation.legacy_job_adapter import experiment_cfgs_from_legacy_eval_config
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


# TODO(cvolk, 2026-07-06): Delete this direct argparse construction helper when
# callers use typed experiment configuration through eval_runner.
def load_env(
    arena_env_args: list[str],
    job_name: str,
    variations: list[str] | None = None,
    render_mode: str | None = None,
    language_instruction: str | None = None,
):

    args_parser = get_isaaclab_arena_environments_cli_parser()

    arena_env_args_cli = args_parser.parse_args(arena_env_args)
    # Optionally override the language instruction.
    arena_env_args_cli.language_instruction = language_instruction
    arena_builder = get_arena_builder_from_cli(arena_env_args_cli, hydra_overrides=variations)

    _, env_cfg, env_kwargs = arena_builder.build_registered()

    # Set unique dataset filename for this job to avoid file locking conflicts
    if env_cfg.recorders is not None:
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


def _run_chunk(chunk_label: str, chunk_jobs: list[dict]) -> int:
    """Run ``chunk_jobs`` in a fresh ``eval_runner`` subprocess and return its exit code."""
    print(f"[eval_runner] {chunk_label}", flush=True)
    # Serialize this chunk's jobs to a temp config the child loads via --eval_jobs_config.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"jobs": chunk_jobs}, tmp)
        chunk_path = Path(tmp.name)
    # Re-run this invocation in the child, with --eval_jobs_config appended so it wins over
    # the master config (argparse keeps the last value).
    # Strip --serve_evaluation_report: a child that served its report would block on
    # serve_until_ctrl_c forever.
    forwarded_args = [arg for arg in sys.argv if arg != "--serve_evaluation_report"]
    config_override = ["--eval_jobs_config", str(chunk_path)]
    child_cmd = [sys.executable, *forwarded_args, *config_override]
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

    if args_cli.serve_evaluation_report:
        print("--serve_evaluation_report is ignored with --chunk_size.", flush=True)

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
        experiment_cfgs = experiment_cfgs_from_legacy_eval_config(
            eval_jobs_config,
            device=args_cli.device,
        )
        experiment_cfgs_by_name = {experiment_cfg.name: experiment_cfg for experiment_cfg in experiment_cfgs}
        # TODO(cvolk, 2026-07-06): Replace JobManager with typed experiment results
        # when the JSON job frontend is removed.
        job_manager = JobManager(eval_jobs_config["jobs"])
        metrics_logger = MetricsLogger()

        job_manager.print_jobs_info()

        # One reverse-dated run directory shared by all jobs; each job gets a subdirectory within it.
        # Always dated so every run produces its own report dir, recording or not.
        # TODO(alexmillane): Currently each chunk produces its own output directory.
        # We should use the same output directory for all chunks in the future.
        run_output_dir = timestamped_run_dir(args_cli.output_base_dir)

        if args_cli.record_viewport_video:
            os.makedirs(run_output_dir, exist_ok=True)
            print(f"[INFO] Video recording enabled. Videos will be saved to: {run_output_dir}")

        for job in job_manager:
            if job is None:
                continue
            experiment_cfg = experiment_cfgs_by_name[job.name]
            job_output_dir = os.path.join(run_output_dir, job.name)
            try:
                result = build_and_run_experiment(
                    experiment_cfg,
                    output_dir=job_output_dir,
                    video_cfg=VideoRecordingCfg(
                        record_viewport_video=args_cli.record_viewport_video,
                        record_camera_video=args_cli.record_camera_video,
                        video_base_dir=job_output_dir,
                    ),
                )
            except Exception as error:
                job_manager.complete_job(job, metrics={}, status=Status.FAILED)
                print(f"Job {job.name} failed with error: {error}")
                print(f"Traceback: {traceback.format_exc()}")
                if not args_cli.continue_on_error:
                    raise
                continue

            status = Status.COMPLETED if result.status is ExperimentStatus.COMPLETED else Status.FAILED
            job_manager.complete_job(job, metrics=result.metrics or {}, status=status)
            if result.metrics is not None:
                metrics_logger.append_job_metrics(job.name, result.metrics)

        job_manager.print_jobs_info()
        metrics_logger.print_metrics()

        # Write HTML report.
        report_path = build_report(run_output_dir)
        if args_cli.serve_evaluation_report:
            serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)


if __name__ == "__main__":
    main()
