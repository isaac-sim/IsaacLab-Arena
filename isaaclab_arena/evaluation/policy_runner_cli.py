# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

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
        default=100,
        help="Number of steps to run the policy for",
    )

