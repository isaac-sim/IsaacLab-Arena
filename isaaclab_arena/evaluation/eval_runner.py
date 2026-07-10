# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment import ArenaExperiment
from isaaclab_arena.evaluation.arena_experiment_config_loader import (
    load_arena_experiment_from_config_file,
    validate_experiment_config_path,
)
from isaaclab_arena.evaluation.arena_run import build_runs_info_table
from isaaclab_arena.evaluation.eval_runner_cli import parse_eval_runner_args
from isaaclab_arena.evaluation.legacy_eval_runner import (
    legacy_json_experiment_requires_cameras,
    load_legacy_json_experiment_config,
    run_legacy_json_in_chunks,
)
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


# TODO(cvolk, 2026-07-10): [typed-config-migration] Typed YAML is composed only
# after SimulationApp starts, so eval_runner cannot determine camera requirements
# in time to configure AppLauncher. Add enable_cameras to ArenaEnvironmentCfg,
# update each factory to honor it or reject unsupported cameras, and apply YAML
# values and Hydra overrides before startup. Enable AppLauncher when any Run enables
# cameras, then remove this getattr and the requirement to also pass --enable_cameras.
def _assert_camera_support_enabled(experiment: ArenaExperiment, enable_cameras: bool) -> None:
    """Check that AppLauncher enabled camera support requested by typed Runs."""
    camera_run_names = [run_cfg.name for run_cfg in experiment if getattr(run_cfg.environment, "enable_cameras", False)]
    assert not camera_run_names or enable_cameras, (
        f"Runs {camera_run_names} enable environment cameras. Pass --enable_cameras so AppLauncher enables "
        "camera support before the typed Experiment is composed."
    )


def main():
    args_cli, experiment_overrides = parse_eval_runner_args()
    experiment_config_path = validate_experiment_config_path(args_cli.experiment_config)
    legacy_experiment_config = load_legacy_json_experiment_config(
        experiment_config_path,
        experiment_overrides,
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
                overrides=experiment_overrides,
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

        if len(legacy_experiment_config["jobs"]) > args_cli.chunk_size:
            run_legacy_json_in_chunks(args_cli, legacy_experiment_config)
            return

    with SimulationAppContext(args_cli):
        experiment = load_arena_experiment_from_config_file(
            experiment_config_path,
            device=args_cli.device,
            overrides=experiment_overrides,
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
