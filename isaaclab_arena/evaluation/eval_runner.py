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
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, build_runs_info_table
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.legacy_eval_config import run_cfgs_from_legacy_eval_config
from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg, execute_experiment
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.video.video_recording import timestamped_run_dir
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c


# TODO(cvolk): Move experiment-level variation inspection out of this CLI entry point.
# Run orchestration belongs in evaluation; catalogue formatting belongs in variations.
def list_variations(run_cfgs: list[ArenaRunCfg]) -> None:
    """Print the Hydra-configurable variations for each run's environment."""
    for run_cfg in run_cfgs:
        arena_builder = build_arena_builder_from_run_cfg(run_cfg)
        print(f"=== Variations for run '{run_cfg.name}' ===", flush=True)
        print(arena_builder.get_variations_catalogue_as_string(), flush=True)


def legacy_json_experiment_requires_cameras(experiment_config: dict) -> bool:
    """Return whether a legacy JSON Experiment requires camera support.

    Args:
        experiment_config: Legacy Experiment mapping containing entries below jobs.

    Returns:
        Whether any legacy entry enables environment cameras.
    """
    return any(
        job_config.get("arena_env_args", {}).get("enable_cameras", False) for job_config in experiment_config["jobs"]
    )


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


def _run_in_chunks(args_cli: argparse.Namespace, experiment_config: dict) -> None:
    """Run each chunk of ``experiment_config['jobs']`` in a fresh subprocess."""
    jobs = experiment_config["jobs"]
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

    # Load the experiment before starting simulation so process-wide requirements
    # can be configured before Isaac Sim starts.
    add_eval_runner_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()
    assert not args_cli.distributed, "Distributed evaluation is not supported yet"

    assert os.path.exists(
        args_cli.eval_jobs_config
    ), f"eval_jobs_config file does not exist: {args_cli.eval_jobs_config}"

    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        experiment_config = json.load(f)

    # Print the variations catalogue for each run's environment and exit.
    if args_cli.list_variations:
        with SimulationAppContext(args_cli):
            run_cfgs = run_cfgs_from_legacy_eval_config(experiment_config, device=args_cli.device)
            list_variations(run_cfgs)
        return

    # Chunked dispatch (--chunk_size N). Splits this config across subprocesses so each
    # gets a fresh SimulationApp. Required for long sweeps because some host memory leaks
    # each cycle and is only reclaimed when the process exits — in-process teardown can't
    # release it.
    if args_cli.chunk_size is not None and len(experiment_config["jobs"]) > args_cli.chunk_size:
        # TODO(cvolk): aggregate per-chunk metrics into one centralized view. Each chunk
        # subprocess currently prints its own MetricsLogger summary and nothing is merged
        # or persisted (save_metrics_to_file() is unused). Follow-up: have each chunk write
        # metrics JSON to a temp file (forward --metrics_file), then merge + print/save here.
        _run_in_chunks(args_cli, experiment_config)
        return

    # Check if any run requires cameras and enable them if needed before starting simulation.
    if args_cli.record_camera_video or legacy_json_experiment_requires_cameras(experiment_config):
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli):
        run_cfgs = run_cfgs_from_legacy_eval_config(
            experiment_config,
            device=args_cli.device,
        )
        metrics_logger = MetricsLogger()

        print(build_runs_info_table(run_cfgs, []))

        # One reverse-dated output directory for the experiment; each legacy job/run
        # gets a subdirectory within it. Always date it so each invocation produces
        # its own report directory, recording or not.
        # TODO(alexmillane): Currently each chunk produces its own output directory.
        # We should use the same output directory for all chunks in the future.
        experiment_output_dir = Path(timestamped_run_dir(args_cli.output_base_dir))

        if args_cli.record_viewport_video:
            os.makedirs(experiment_output_dir, exist_ok=True)
            print(f"[INFO] Video recording enabled. Videos will be saved to: {experiment_output_dir}")

        results = execute_experiment(
            run_cfgs,
            output_dir=experiment_output_dir,
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            continue_on_error=args_cli.continue_on_error,
        )
        for result in results:
            if result.metrics is not None:
                metrics_logger.append_job_metrics(result.run_name, result.metrics)

        print(build_runs_info_table(run_cfgs, results))
        metrics_logger.print_metrics()

        # Write HTML report.
        report_path = build_report(experiment_output_dir)
        if args_cli.serve_evaluation_report:
            serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)


if __name__ == "__main__":
    main()
