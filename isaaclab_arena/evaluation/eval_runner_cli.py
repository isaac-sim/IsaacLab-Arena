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
        default="isaaclab_arena/evaluation/configs/gr00t_jobs_config.json",
        help="Path to the eval jobs config file.",
    )
