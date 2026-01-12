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

    remote_group = parser.add_argument_group(
    "Remote Policy",
    "Arguments for remote policy deployment.",
    )
    
    remote_group.add_argument(
        "--remote_host",
        type=str,
        default=None,
        help="Remote policy server host. If not set, the policy will be treated as local-only.",
    )

    remote_group.add_argument(
        "--remote_port",
        type=int,
        default=5555,
        help="Remote policy server port.",
    )

    remote_group.add_argument(
        "--remote_api_token",
        type=str,
        default=None,
        help="Optional API token for the remote policy server.",
    )

    remote_group.add_argument(
        "--remote_timeout_ms",
        type=int,
        default=5000,
        help="Timeout in milliseconds for remote policy calls.",
    )

    remote_group.add_argument(
        "--remote_kill_on_exit",
        action="store_true",
        help="If set, send a 'kill' request to the remote policy server when the run finishes.",
    )
