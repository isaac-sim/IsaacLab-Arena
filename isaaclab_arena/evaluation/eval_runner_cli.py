# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse


def add_eval_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add eval runner specific arguments to the parser."""
    parser.add_argument(
        "--eval_jobs_config",
        type=str,
        default="isaaclab_arena_environments/eval_jobs_configs/zero_action_jobs_config.json",
        help="Path to the eval jobs config file.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record videos for each eval job.")
    parser.add_argument(
        "--camera_video",
        action="store_true",
        default=False,
        help="Record one mp4 per (env, camera, episode) from obs['camera_obs'] for each eval job.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/eval/videos",
        help="Root directory for recorded videos. Each job gets a subdirectory.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Continue evaluation with remaining jobs when a job fails instead of stopping immediately.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Run jobs in chunks of at most this many, one fresh subprocess per chunk."
            " Each restart lets the OS reclaim accumulated memory, avoiding OOM on"
            " long sweeps. Default unset — single process. Leave unset for normal runs;"
            " set only if a long sweep grows in host memory or gets OOM-killed."
        ),
    )
