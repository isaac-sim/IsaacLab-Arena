# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides

_DEFAULT_EXPERIMENT_CONFIG_PATH = "isaaclab_arena_environments/eval_jobs_configs/zero_action_jobs_config.json"


def add_eval_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add eval runner specific arguments to the parser."""
    # TODO(cvolk, 2026-07-09): [typed-config-migration] Remove the --eval_jobs_config alias
    # when legacy JSON Experiments are retired. Both flags currently populate experiment_config.
    parser.add_argument(
        "--experiment_config",
        "--eval_jobs_config",
        dest="experiment_config",
        type=str,
        default=_DEFAULT_EXPERIMENT_CONFIG_PATH,
        help=(
            "Path to a typed YAML Experiment or legacy JSON evaluation config. "
            "For YAML, append Hydra KEY=VALUE overrides to the command."
        ),
    )
    parser.add_argument(
        "--record_viewport_video",
        action="store_true",
        default=False,
        help="Record viewport videos for each Run.",
    )
    parser.add_argument(
        "--record_camera_video",
        action="store_true",
        default=False,
        help="Record one mp4 per (env, camera, episode) from obs['camera_obs'] for each Run.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/eval/output",
        help=(
            "Base directory for evaluation outputs (videos, per-episode results, report); a"
            " reverse-dated Experiment subdirectory and per-Run subdirectory are added."
        ),
    )
    parser.add_argument(
        "--serve_evaluation_report",
        action="store_true",
        default=False,
        help="After all Runs finish, serve the evaluation report over HTTP.",
    )
    parser.add_argument(
        "--evaluation_report_port",
        type=int,
        default=8000,
        help="Port to serve the evaluation report on when --serve_evaluation_report is set. Defaults to 8000.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Continue evaluation with remaining Runs when a Run fails instead of stopping immediately.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Run legacy JSON entries in chunks of at most this many, one fresh subprocess per chunk."
            " Each restart lets the OS reclaim accumulated memory, avoiding OOM on"
            " long sweeps. Default unset — single process. Leave unset for normal runs;"
            " set only if a long sweep grows in host memory or gets OOM-killed."
        ),
    )


def parse_eval_runner_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse eval-runner arguments and return native Hydra override tokens separately.

    Args:
        argv: Arguments to parse, or None to read the process arguments.
    """
    parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(parser)
    parser.allow_abbrev = False
    args_cli, experiment_overrides = parser.parse_known_args(argv)
    assert_hydra_overrides(experiment_overrides, parser)
    assert not args_cli.distributed, "Distributed evaluation is not supported yet"
    return args_cli, experiment_overrides
