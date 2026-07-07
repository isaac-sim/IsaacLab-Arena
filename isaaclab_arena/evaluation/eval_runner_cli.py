# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse


def add_eval_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add eval runner specific arguments to the parser."""
    parser.add_argument(
        "experiment_config",
        type=str,
        nargs="?",
        help="YAML experiment collection to evaluate.",
    )
    # TODO(cvolk, 2026-07-07): Remove this alias after the remaining eval-jobs
    # JSON documents migrate to typed YAML experiment collections.
    parser.add_argument(
        "--eval_jobs_config",
        dest="legacy_eval_jobs_config",
        type=str,
        default=None,
        help="Deprecated path to a legacy JSON eval-jobs configuration.",
    )
    parser.add_argument(
        "--record_viewport_video",
        action="store_true",
        default=False,
        help="Record viewport videos for each experiment.",
    )
    parser.add_argument(
        "--record_camera_video",
        action="store_true",
        default=False,
        help="Record one mp4 per (env, camera, episode) from obs['camera_obs'] for each experiment.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/eval/output",
        help=(
            "Base directory for evaluation outputs (videos, per-episode results, report); a"
            " reverse-dated run subdirectory and per-experiment subdirectory are added."
        ),
    )
    parser.add_argument(
        "--serve_evaluation_report",
        action="store_true",
        default=False,
        help="After all experiments finish, serve the evaluation report over HTTP.",
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
        help="Continue with remaining experiments when one fails instead of stopping immediately.",
    )
