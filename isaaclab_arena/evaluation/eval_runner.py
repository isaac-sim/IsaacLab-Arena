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
from isaaclab_arena.evaluation.arena_experiment import ArenaExperiment
from isaaclab_arena.evaluation.arena_experiment_config_loader import (
    load_arena_experiment_from_config_file,
    validate_experiment_config_path,
)
from isaaclab_arena.evaluation.arena_run import build_runs_info_table
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg, execute_experiment
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.video.video_recording import timestamped_run_dir
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c


# TODO(cvolk): Move experiment-level variation inspection out of this CLI entry point.
# Run orchestration belongs in evaluation; catalogue formatting belongs in variations.
def list_variations(experiment: ArenaExperiment) -> None:
    """Print the Hydra-configurable variations for each run's environment."""
    for run_cfg in experiment:
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


def _assert_camera_support_enabled(experiment: ArenaExperiment, enable_cameras: bool) -> None:
    """Check that AppLauncher enabled camera support requested by typed Runs."""
    camera_run_names = [run_cfg.name for run_cfg in experiment if getattr(run_cfg.environment, "enable_cameras", False)]
    assert not camera_run_names or enable_cameras, (
        f"Runs {camera_run_names} enable environment cameras. Pass --enable_cameras so AppLauncher enables "
        "camera support before the typed Experiment is composed."
    )


def _run_legacy_json_chunk(chunk_label: str, legacy_job_configs: list[dict]) -> int:
    """Run legacy JSON entries in a fresh eval_runner subprocess."""
    print(f"[eval_runner] {chunk_label}", flush=True)
    # Serialize this chunk's jobs to a temp config the child loads via --experiment_config.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"jobs": legacy_job_configs}, tmp)
        chunk_path = Path(tmp.name)
    # Re-run this invocation in the child, with --experiment_config appended so it wins over
    # the master config (argparse keeps the last value).
    # Strip --serve_evaluation_report: a child that served its report would block on
    # serve_until_ctrl_c forever.
    forwarded_args = [arg for arg in sys.argv if arg != "--serve_evaluation_report"]
    config_override = ["--experiment_config", str(chunk_path)]
    child_cmd = [sys.executable, *forwarded_args, *config_override]
    try:
        result = subprocess.run(child_cmd, check=False)
    finally:
        # Remove the temp chunk config now that the child has loaded it.
        chunk_path.unlink(missing_ok=True)
    return result.returncode


def _run_legacy_json_in_chunks(args_cli: argparse.Namespace, legacy_experiment_config: dict) -> None:
    """Run each chunk of a legacy JSON Experiment in a fresh subprocess."""
    jobs = legacy_experiment_config["jobs"]
    chunk_size = args_cli.chunk_size
    assert chunk_size > 0, f"--chunk_size must be positive, got {chunk_size}"
    n_chunks = math.ceil(len(jobs) / chunk_size)
    print(f"[eval_runner] {len(jobs)} Runs → {n_chunks} chunks of <= {chunk_size}", flush=True)

    if args_cli.serve_evaluation_report:
        print("--serve_evaluation_report is ignored with --chunk_size.", flush=True)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(jobs))
        chunk_label = f"chunk {chunk_idx + 1}/{n_chunks}: Runs {start}..{end - 1}"
        returncode = _run_legacy_json_chunk(chunk_label, jobs[start:end])
        if returncode != 0:
            print(f"[eval_runner] chunk {chunk_idx} failed (exit {returncode}).", flush=True)
            sys.exit(returncode)


def _parse_eval_runner_args() -> argparse.Namespace:
    """Parse and validate the eval runner command-line arguments."""
    args_parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()
    assert not args_cli.distributed, "Distributed evaluation is not supported yet"
    return args_cli


def _load_legacy_json_experiment_config(
    experiment_config_path: Path,
    experiment_overrides: list[str],
) -> dict | None:
    """Load legacy JSON data needed before startup, or return None for YAML.

    Args:
        experiment_config_path: Validated path to an Experiment configuration.
        experiment_overrides: Hydra overrides supplied for a typed YAML Experiment.

    Returns:
        The legacy JSON mapping, or None when the Experiment uses YAML.
    """
    if experiment_config_path.suffix.lower() != ".json":
        return None

    assert not experiment_overrides, "Experiment overrides are supported only for typed YAML Experiments"
    with experiment_config_path.open(encoding="utf-8") as experiment_config_file:
        return json.load(experiment_config_file)


def main():
    args_cli = _parse_eval_runner_args()
    experiment_config_path = validate_experiment_config_path(args_cli.experiment_config)
    legacy_experiment_config = _load_legacy_json_experiment_config(
        experiment_config_path,
        args_cli.experiment_override,
    )

    if args_cli.record_camera_video or (
        legacy_experiment_config is not None and legacy_json_experiment_requires_cameras(legacy_experiment_config)
    ):
        args_cli.enable_cameras = True

    # Print the variations catalogue for each run's environment and exit.
    if args_cli.list_variations:
        with SimulationAppContext(args_cli):
            experiment = load_arena_experiment_from_config_file(
                experiment_config_path,
                device=args_cli.device,
                overrides=args_cli.experiment_override,
            )
            _assert_camera_support_enabled(experiment, args_cli.enable_cameras)
            list_variations(experiment)
        return

    # Chunked dispatch (--chunk_size N). Splits this config across subprocesses so each
    # gets a fresh SimulationApp. Required for long sweeps because some host memory leaks
    # each cycle and is only reclaimed when the process exits — in-process teardown can't
    # release it.
    if args_cli.chunk_size is not None:
        assert legacy_experiment_config is not None, "--chunk_size currently supports only legacy JSON Experiments"

        # TODO(cvolk): aggregate per-chunk metrics into one centralized view. Each chunk
        # subprocess currently prints its own MetricsLogger summary and nothing is merged
        # or persisted (save_metrics_to_file() is unused). Follow-up: have each chunk write
        # metrics JSON to a temp file (forward --metrics_file), then merge + print/save here.
        if len(legacy_experiment_config["jobs"]) > args_cli.chunk_size:
            _run_legacy_json_in_chunks(args_cli, legacy_experiment_config)
            return

    with SimulationAppContext(args_cli):
        experiment = load_arena_experiment_from_config_file(
            experiment_config_path,
            device=args_cli.device,
            overrides=args_cli.experiment_override,
        )
        _assert_camera_support_enabled(experiment, args_cli.enable_cameras)
        metrics_logger = MetricsLogger()

        print(build_runs_info_table(experiment, []))

        # One reverse-dated output directory for the Experiment, with one subdirectory
        # per Run. Always date it so each invocation produces its own report directory.
        # TODO(alexmillane): Currently each chunk produces its own output directory.
        # We should use the same output directory for all chunks in the future.
        experiment_output_dir = Path(timestamped_run_dir(args_cli.output_base_dir))

        if args_cli.record_viewport_video:
            os.makedirs(experiment_output_dir, exist_ok=True)
            print(f"[INFO] Video recording enabled. Videos will be saved to: {experiment_output_dir}")

        results = execute_experiment(
            experiment,
            output_dir=experiment_output_dir,
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            continue_on_error=args_cli.continue_on_error,
        )
        for result in results:
            if result.metrics is not None:
                metrics_logger.append_job_metrics(result.run_name, result.metrics)

        print(build_runs_info_table(experiment, results))
        metrics_logger.print_metrics()

        # Write HTML report.
        report_path = build_report(experiment_output_dir)
        if args_cli.serve_evaluation_report:
            serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)


if __name__ == "__main__":
    main()
