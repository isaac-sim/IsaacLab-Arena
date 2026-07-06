# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dispatch typed Arena experiments from the temporary eval-jobs JSON frontend."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from prettytable import PrettyTable

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, ArenaExperimentResult, ExperimentStatus
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.experiment_runner import ArenaBuilderFactory, build_arena_builder, run_experiment
from isaaclab_arena.evaluation.legacy_job_config import (
    LegacyCliEnvironmentCfg,
    arena_experiments_from_legacy_config,
    build_legacy_cli_arena_builder,
)
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c


def list_variations(experiments: list[ArenaExperimentCfg]) -> None:
    """Print the Hydra-configurable variations for every experiment."""
    for experiment in experiments:
        arena_builder = _arena_builder_factory_for(experiment)(experiment)
        print(f"=== Variations for experiment '{experiment.name}' ===", flush=True)
        print(arena_builder.get_variations_catalogue_as_string(), flush=True)


# TODO(cvolk, 2026-07-06): Remove graph-specific builder selection when graph
# environments participate in the typed environment-factory contract.
def _arena_builder_factory_for(experiment: ArenaExperimentCfg) -> ArenaBuilderFactory:
    """Select typed construction or the isolated legacy graph adapter."""
    if isinstance(experiment.environment, LegacyCliEnvironmentCfg):
        return build_legacy_cli_arena_builder
    return build_arena_builder


# TODO(cvolk, 2026-07-06): Delete this raw-dictionary inspection when the YAML
# frontend composes typed experiments before configuring SimulationApp.
def enable_cameras_if_required(eval_jobs_config: dict, args_cli: argparse.Namespace) -> None:
    """Enable process-wide camera support when a legacy job requests cameras."""
    for job_config in eval_jobs_config["jobs"]:
        if job_config.get("arena_env_args", {}).get("enable_cameras", False):
            args_cli.enable_cameras = True
            return


# TODO(cvolk, 2026-07-06): Replace temporary JSON chunk documents with typed YAML
# experiment dispatch when the YAML frontend is introduced.
def _run_chunk(chunk_label: str, chunk_jobs: list[dict]) -> int:
    """Run legacy ``chunk_jobs`` in a fresh eval-runner subprocess."""
    print(f"[eval_runner] {chunk_label}", flush=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temporary_file:
        json.dump({"jobs": chunk_jobs}, temporary_file)
        chunk_path = Path(temporary_file.name)

    forwarded_args = [argument for argument in sys.argv if argument != "--serve_evaluation_report"]
    child_command = [sys.executable, *forwarded_args, "--eval_jobs_config", str(chunk_path)]
    try:
        result = subprocess.run(child_command, check=False)
    finally:
        chunk_path.unlink(missing_ok=True)
    return result.returncode


def _run_in_chunks(args_cli: argparse.Namespace, legacy_config: dict) -> None:
    """Run each legacy job chunk in a fresh eval-runner subprocess."""
    jobs = legacy_config["jobs"]
    chunk_size = args_cli.chunk_size
    assert chunk_size > 0, f"--chunk_size must be positive, got {chunk_size}"
    num_chunks = math.ceil(len(jobs) / chunk_size)
    print(f"[eval_runner] {len(jobs)} jobs → {num_chunks} chunks of <= {chunk_size}", flush=True)

    if args_cli.serve_evaluation_report:
        print("--serve_evaluation_report is ignored with --chunk_size.", flush=True)

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(jobs))
        chunk_label = f"chunk {chunk_index + 1}/{num_chunks}: jobs {start}..{end - 1}"
        return_code = _run_chunk(chunk_label, jobs[start:end])
        if return_code != 0:
            print(f"[eval_runner] chunk {chunk_index} failed (exit {return_code}).", flush=True)
            sys.exit(return_code)


def evaluate_experiments(
    args_cli: argparse.Namespace,
    experiments: list[ArenaExperimentCfg],
) -> list[ArenaExperimentResult]:
    """Run typed experiments sequentially inside the active simulation application."""
    # TODO(cvolk, 2026-07-06): Replace this legacy argparse dispatcher input with
    # a typed run configuration when the YAML frontend is introduced.
    assert experiments, "evaluation must contain at least one experiment"
    experiment_names = [experiment.name for experiment in experiments]
    assert len(experiment_names) == len(set(experiment_names)), "experiment names must be unique"

    results: list[ArenaExperimentResult] = []
    metrics_logger = MetricsLogger()
    _print_experiments_info(experiments, results)

    # One output directory is shared by every sequential experiment in this process.
    # TODO(alexmillane): Currently each chunk produces its own output directory.
    # We should use the same output directory for all chunks in the future.
    run_output_dir = timestamped_run_dir(args_cli.output_base_dir)
    if args_cli.record_viewport_video:
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"[INFO] Video recording enabled. Videos will be saved to: {run_output_dir}")

    for experiment in experiments:
        print(f"Running experiment {experiment.name}")
        experiment_output_dir = os.path.join(run_output_dir, experiment.name)
        result = run_experiment(
            experiment,
            output_dir=experiment_output_dir,
            video_cfg=VideoRecordingCfg(
                record_viewport_video=args_cli.record_viewport_video,
                record_camera_video=args_cli.record_camera_video,
                video_base_dir=experiment_output_dir,
            ),
            arena_builder_factory=_arena_builder_factory_for(experiment),
        )
        results.append(result)

        if result.metrics is not None:
            metrics_logger.append_experiment_metrics(experiment.name, result.metrics)

        if result.status is ExperimentStatus.FAILED:
            print(f"Experiment {experiment.name} failed:\n{result.error}")
            if not args_cli.continue_on_error:
                raise RuntimeError(f"Experiment {experiment.name} failed:\n{result.error}")

    _print_experiments_info(experiments, results)
    metrics_logger.print_metrics()

    report_path = build_report(run_output_dir)
    if args_cli.serve_evaluation_report:
        serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)
    return results


def _print_experiments_info(
    experiments: list[ArenaExperimentCfg],
    results: list[ArenaExperimentResult],
) -> None:
    """Print the configured experiments and their latest execution status."""
    results_by_name = {result.experiment_name: result for result in results}
    table = PrettyTable(
        field_names=[
            "Experiment Name",
            "Status",
            "Policy Type",
            "Num Envs",
            "Num Steps",
            "Num Episodes",
            "Num Rebuilds",
        ]
    )
    for experiment in experiments:
        result = results_by_name.get(experiment.name)
        policy_type = PolicyRegistry().get_policy_type_for_cfg(experiment.policy)
        table.add_row([
            experiment.name,
            result.status.value if result is not None else "pending",
            policy_type.name,
            experiment.environment_builder.num_envs,
            experiment.rollout.num_steps,
            experiment.rollout.num_episodes,
            experiment.num_rebuilds,
        ])
    print(table)


def main() -> None:
    """Load the legacy JSON frontend and dispatch its typed experiments."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.parse_known_args()

    add_eval_runner_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()
    assert not args_cli.distributed, "Distributed evaluation is not supported yet"
    assert os.path.exists(
        args_cli.eval_jobs_config
    ), f"eval_jobs_config file does not exist: {args_cli.eval_jobs_config}"

    # TODO(cvolk, 2026-07-06): Delete JSON loading when evaluation configuration
    # moves to the structured YAML experiment frontend.
    with open(args_cli.eval_jobs_config, encoding="utf-8") as config_file:
        legacy_config = json.load(config_file)

    if args_cli.record_camera_video:
        args_cli.enable_cameras = True
    enable_cameras_if_required(legacy_config, args_cli)

    if args_cli.list_variations:
        with SimulationAppContext(args_cli):
            experiments = arena_experiments_from_legacy_config(legacy_config, device=args_cli.device)
            list_variations(experiments)
        return

    if args_cli.chunk_size is not None and len(legacy_config["jobs"]) > args_cli.chunk_size:
        # TODO(cvolk, 2026-07-06): Aggregate per-chunk metrics into one centralized
        # result when the dispatcher stops communicating through temporary config files.
        _run_in_chunks(args_cli, legacy_config)
        return

    with SimulationAppContext(args_cli):
        experiments = arena_experiments_from_legacy_config(legacy_config, device=args_cli.device)
        evaluate_experiments(args_cli, experiments)


if __name__ == "__main__":
    main()
