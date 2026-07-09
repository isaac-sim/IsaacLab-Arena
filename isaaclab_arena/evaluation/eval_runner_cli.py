# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

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
        help="Path to a typed YAML Experiment or legacy JSON evaluation config.",
    )
    parser.add_argument(
        "--experiment_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Hydra override for a typed YAML Experiment. Repeat for multiple overrides. "
            "Also pass --enable_cameras when an override enables environment cameras."
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
