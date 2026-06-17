# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse


def add_policy_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add policy runner specific arguments to the parser."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Type of policy to use. This is either a registered policy name or a path to a policy class.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to run the policy (if num_episodes is not provided)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run the policy (if num_steps is not provided)",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default=None,
        help="Language instruction for the policy. Takes precedence over the task's own description.",
    )
    parser.add_argument(
        "--record_viewport_video",
        action="store_true",
        default=False,
        help="Record an mp4 video of the rollout viewport (uses gymnasium.wrappers.RecordVideo).",
    )
    parser.add_argument(
        "--video_base_dir",
        type=str,
        default="/eval/videos",
        help=(
            "Base directory for recorded videos; a reverse-dated run subdirectory is added per run."
            " Used with --record_viewport_video and/or --record_camera_video."
        ),
    )
    parser.add_argument(
        "--record_camera_video",
        action="store_true",
        default=False,
        help=(
            "Record one mp4 per camera in obs['camera_obs'] (what the policy actually sees)."
            " Independent of --record_viewport_video; use either or both."
        ),
    )
    parser.add_argument(
        "--serve_evaluation_report",
        action="store_true",
        default=False,
        help="After all jobs finish, serve the evaluation report over HTTP.",
    )
    parser.add_argument(
        "--evaluation_report_port",
        type=int,
        default=8000,
        help="Port to serve the evaluation report on when --serve_evaluation_report is set. Defaults to 8000.",
    )
