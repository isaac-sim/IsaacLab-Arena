# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run an Arena Experiment locally or submit it to OSMO."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.evaluation.experiment_runner_route import (
    parse_experiment_runner_route,
    print_experiment_runner_help,
    should_show_experiment_runner_help,
    uses_experiment_runner_cfg,
)

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_experiment import ArenaExperiment
    from isaaclab_arena.evaluation.experiment_runner_cfg import ExperimentRunnerCfg


def list_variations(experiment: ArenaExperiment) -> None:
    """Print the Hydra-configurable variations for each run's environment."""
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg

    for run_cfg in experiment:
        arena_builder = build_arena_builder_from_run_cfg(run_cfg)
        print(f"=== Variations for run '{run_cfg.name}' ===", flush=True)
        print(arena_builder.get_variations_catalogue_as_string(), flush=True)


# TODO(cvolk, 2026-07-10): [typed-config-migration] Typed YAML is composed only
# after SimulationApp starts, so the Experiment Runner cannot determine camera requirements
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


def _get_simulation_app_context_type() -> Any:
    """Import the local SimulationApp context only after local execution is selected."""
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext as context_type

    return context_type


def _get_experiment_loader() -> Any:
    """Import the typed Experiment loader only after local execution is selected."""
    from isaaclab_arena.evaluation.arena_experiment_config_loader import (
        load_arena_experiment_from_config_file as experiment_loader,
    )

    return experiment_loader


def _execute_experiment_and_report(
    experiment: ArenaExperiment,
    *,
    output_base_dir: str,
    record_viewport_video: bool,
    record_camera_video: bool,
    continue_on_error: bool,
    serve_evaluation_report: bool,
    evaluation_report_port: int,
) -> None:
    """Execute a loaded Experiment and produce its metrics and report."""
    from isaaclab_arena.evaluation.arena_run import build_runs_info_table
    from isaaclab_arena.evaluation.run_execution import execute_experiment
    from isaaclab_arena.metrics.metrics_logger import MetricsLogger
    from isaaclab_arena.video.video_recording import timestamped_run_dir
    from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c

    metrics_logger = MetricsLogger()
    print(build_runs_info_table(experiment, []))

    # One reverse-dated output directory for the Experiment, with one subdirectory
    # per Run. Always date it so each invocation produces its own report directory.
    experiment_output_dir = Path(timestamped_run_dir(output_base_dir))
    if record_viewport_video:
        os.makedirs(experiment_output_dir, exist_ok=True)
        print(f"[INFO] Video recording enabled. Videos will be saved to: {experiment_output_dir}")

    results = execute_experiment(
        experiment,
        output_dir=experiment_output_dir,
        record_viewport_video=record_viewport_video,
        record_camera_video=record_camera_video,
        continue_on_error=continue_on_error,
    )
    for result in results:
        if result.metrics is not None:
            metrics_logger.append_job_metrics(result.run_name, result.metrics)

    print(build_runs_info_table(experiment, results))
    metrics_logger.print_metrics()

    report_path = build_report(experiment_output_dir)
    if serve_evaluation_report:
        serve_until_ctrl_c(report_path.parent, evaluation_report_port, report_path.name)


def _run_experiment_runner_cfg_locally(cfg: ExperimentRunnerCfg, app_launcher_args: argparse.Namespace) -> int:
    """Execute one Experiment Runner configuration in the current container."""
    if cfg.record_camera_video:
        app_launcher_args.enable_cameras = True

    simulation_app_context_type = _get_simulation_app_context_type()
    experiment_loader = _get_experiment_loader()
    with simulation_app_context_type(app_launcher_args):
        experiment = experiment_loader(
            cfg.experiment_config,
            device=app_launcher_args.device,
            overrides=cfg.experiment_overrides,
        )
        _assert_camera_support_enabled(experiment, app_launcher_args.enable_cameras)
        if app_launcher_args.list_variations:
            list_variations(experiment)
            return 0
        _execute_experiment_and_report(
            experiment,
            output_base_dir=cfg.output_base_dir,
            record_viewport_video=cfg.record_viewport_video,
            record_camera_video=cfg.record_camera_video,
            continue_on_error=cfg.continue_on_error,
            serve_evaluation_report=cfg.serve_evaluation_report,
            evaluation_report_port=cfg.evaluation_report_port,
        )
    return 0


def _load_experiment_runner_cfg(config_path: str) -> ExperimentRunnerCfg:
    """Load the typed execution config without importing local runtime modules."""
    from isaaclab_arena.evaluation.experiment_runner_cfg import load_experiment_runner_cfg

    return load_experiment_runner_cfg(config_path)


def _parse_local_app_launcher_args(cli_args: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse AppLauncher arguments only after the local route is selected."""
    from isaaclab_arena.evaluation.experiment_runner_local_cli import parse_local_experiment_runner_args

    return parse_local_experiment_runner_args(cli_args)


def _run_experiment_runner_cfg_route(cli_args: list[str]) -> int:
    """Run the typed Experiment Runner route."""
    route, backend_args = parse_experiment_runner_route(cli_args)
    cfg = _load_experiment_runner_cfg(route.config_path)

    if route.local:
        app_launcher_args, trailing_experiment_overrides = _parse_local_app_launcher_args(backend_args)
    else:
        trailing_experiment_overrides = backend_args

    cfg = replace(
        cfg,
        experiment_overrides=[*cfg.experiment_overrides, *trailing_experiment_overrides],
    )
    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides

    parser = argparse.ArgumentParser(add_help=False)
    assert_hydra_overrides(cfg.experiment_overrides, parser)
    if route.local:
        return _run_experiment_runner_cfg_locally(cfg, app_launcher_args)

    from osmo.experiment_dispatcher import submit_experiment_to_osmo

    return submit_experiment_to_osmo(cfg, route.osmo_config_path)


def _run_legacy_experiment_runner(cli_args: list[str]) -> int:
    """Run the deprecated argparse/Namespace evaluation path unchanged."""
    from isaaclab_arena.evaluation.arena_experiment_config_loader import validate_experiment_config_path
    from isaaclab_arena.evaluation.legacy_experiment_runner import (
        legacy_json_experiment_requires_cameras,
        load_legacy_json_experiment_config,
        run_legacy_json_in_chunks,
    )
    from isaaclab_arena.evaluation.legacy_experiment_runner_cli import parse_legacy_experiment_runner_args

    args_cli, experiment_overrides = parse_legacy_experiment_runner_args(cli_args)
    experiment_config_path = validate_experiment_config_path(args_cli.experiment_config)
    legacy_experiment_config = load_legacy_json_experiment_config(
        experiment_config_path,
        experiment_overrides,
    )

    if args_cli.record_camera_video or (
        legacy_experiment_config is not None and legacy_json_experiment_requires_cameras(legacy_experiment_config)
    ):
        args_cli.enable_cameras = True

    simulation_app_context_type = _get_simulation_app_context_type()
    experiment_loader = _get_experiment_loader()
    if args_cli.list_variations:
        with simulation_app_context_type(args_cli):
            experiment = experiment_loader(
                experiment_config_path,
                device=args_cli.device,
                overrides=experiment_overrides,
            )
            _assert_camera_support_enabled(experiment, args_cli.enable_cameras)
            list_variations(experiment)
        return 0

    if args_cli.chunk_size is not None:
        assert legacy_experiment_config is not None, "--chunk_size currently supports only legacy JSON Experiments"
        if len(legacy_experiment_config["jobs"]) > args_cli.chunk_size:
            run_legacy_json_in_chunks(args_cli, legacy_experiment_config)
            return 0

    with simulation_app_context_type(args_cli):
        experiment = experiment_loader(
            experiment_config_path,
            device=args_cli.device,
            overrides=experiment_overrides,
        )
        _assert_camera_support_enabled(experiment, args_cli.enable_cameras)
        _execute_experiment_and_report(
            experiment,
            output_base_dir=args_cli.output_base_dir,
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            continue_on_error=args_cli.continue_on_error,
            serve_evaluation_report=args_cli.serve_evaluation_report,
            evaluation_report_port=args_cli.evaluation_report_port,
        )
    return 0


def main(cli_args: list[str] | None = None) -> int:
    """Run an Arena Experiment locally or submit it to OSMO."""
    args = list(sys.argv[1:] if cli_args is None else cli_args)
    if should_show_experiment_runner_help(args):
        print_experiment_runner_help()
        return 0
    if uses_experiment_runner_cfg(args):
        return _run_experiment_runner_cfg_route(args)
    return _run_legacy_experiment_runner(args)


if __name__ == "__main__":
    raise SystemExit(main())
