# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed command-line boundary for local Experiment evaluation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides

_DEFAULT_EXPERIMENT_CONFIG_PATH = Path("isaaclab_arena_environments/eval_jobs_configs/zero_action_jobs_config.json")


@dataclass
class EvalRunnerCfg:
    """Typed configuration for one local eval-runner invocation."""

    experiment_config: Path
    """Typed YAML Experiment or deprecated JSON Experiment path."""

    experiment_overrides: list[str]
    """Native Hydra overrides applied to a typed YAML Experiment."""

    app_launcher_args: dict[str, Any]
    """Arguments consumed by Isaac Lab's AppLauncher."""

    invocation_args: list[str]
    """Original arguments forwarded when legacy JSON chunks restart the runner."""

    record_viewport_video: bool = False
    """Whether to record the Kit viewport for every Run."""

    record_camera_video: bool = False
    """Whether to record camera observations for every Run."""

    output_base_dir: Path = Path("/eval/output")
    """Base directory for Experiment outputs."""

    serve_evaluation_report: bool = False
    """Whether to serve the generated report after evaluation."""

    evaluation_report_port: int = 8000
    """Port used by the evaluation report server."""

    continue_on_error: bool = False
    """Whether later Runs continue after one Run fails."""

    chunk_size: int | None = None
    """Deprecated JSON-only subprocess chunk size."""

    list_variations: bool = False
    """Whether to print each Run's variation catalogue and exit."""

    @property
    def device(self) -> str:
        """Return the process-wide simulation device."""
        return str(self.app_launcher_args["device"])


def parse_eval_runner_cfg(cli_args: list[str] | None = None) -> EvalRunnerCfg:
    """Parse local eval-runner arguments into typed configuration.

    Args:
        cli_args: Command-line arguments excluding the program name.

    Returns:
        Typed local evaluation configuration.
    """
    arguments = list(sys.argv[1:] if cli_args is None else cli_args)
    parser = argparse.ArgumentParser(
        description="Run an Isaac Lab-Arena Experiment locally.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--experiment-config",
        "--experiment_config",
        "--eval-jobs-config",
        "--eval_jobs_config",
        dest="experiment_config",
        type=Path,
        default=_DEFAULT_EXPERIMENT_CONFIG_PATH,
        help="Path to a typed YAML Experiment or deprecated JSON evaluation config.",
    )
    parser.add_argument(
        "--record-viewport-video",
        "--record_viewport_video",
        dest="record_viewport_video",
        action="store_true",
        help="Record viewport video for each Run.",
    )
    parser.add_argument(
        "--record-camera-video",
        "--record_camera_video",
        dest="record_camera_video",
        action="store_true",
        help="Record camera-observation video for each Run.",
    )
    parser.add_argument(
        "--output-base-dir",
        "--output_base_dir",
        dest="output_base_dir",
        type=Path,
        default=Path("/eval/output"),
        help="Base directory for Experiment outputs.",
    )
    parser.add_argument(
        "--serve-evaluation-report",
        "--serve_evaluation_report",
        dest="serve_evaluation_report",
        action="store_true",
        help="Serve the generated report after all Runs finish.",
    )
    parser.add_argument(
        "--evaluation-report-port",
        "--evaluation_report_port",
        dest="evaluation_report_port",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--continue-on-error",
        "--continue_on_error",
        dest="continue_on_error",
        action="store_true",
        help="Continue with remaining Runs after a Run fails.",
    )
    parser.add_argument(
        "--chunk-size",
        "--chunk_size",
        dest="chunk_size",
        type=int,
        default=None,
        help="Deprecated JSON-only subprocess chunk size.",
    )
    parser.add_argument(
        "--list-variations",
        "--list_variations",
        dest="list_variations",
        action="store_true",
        help="Print Run variation catalogues and exit.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Unsupported compatibility option; eval_runner does not run distributed.",
    )

    action_ids_before_app_launcher = {id(action) for action in parser._actions}
    AppLauncher.add_app_launcher_args(parser)
    app_launcher_dests = {action.dest for action in parser._actions if id(action) not in action_ids_before_app_launcher}

    parsed, experiment_overrides = parser.parse_known_args(arguments)
    assert_hydra_overrides(experiment_overrides, parser)
    if parsed.distributed:
        parser.error("--distributed is not supported by eval_runner")

    app_launcher_args = {dest: getattr(parsed, dest) for dest in app_launcher_dests}
    app_launcher_args.update({name: value for name, value in vars(parsed).items() if name.endswith("_explicit")})
    return EvalRunnerCfg(
        experiment_config=parsed.experiment_config,
        experiment_overrides=experiment_overrides,
        app_launcher_args=app_launcher_args,
        invocation_args=arguments,
        record_viewport_video=parsed.record_viewport_video,
        record_camera_video=parsed.record_camera_video,
        output_base_dir=parsed.output_base_dir,
        serve_evaluation_report=parsed.serve_evaluation_report,
        evaluation_report_port=parsed.evaluation_report_port,
        continue_on_error=parsed.continue_on_error,
        chunk_size=parsed.chunk_size,
        list_variations=parsed.list_variations,
    )
