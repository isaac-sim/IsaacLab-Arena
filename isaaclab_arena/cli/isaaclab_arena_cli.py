# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab.app import AppLauncher


def get_isaaclab_arena_cli_parser() -> argparse.ArgumentParser:
    """Get a complete argument parser with both Isaac Lab and IsaacLab Arena arguments."""
    parser = argparse.ArgumentParser(description="IsaacLab Arena CLI parser.")
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_external_environments_cli_args(parser)
    return parser


def add_isaac_lab_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add Isaac Lab specific command line arguments to the given parser."""

    isaac_lab_group = parser.add_argument_group("Isaac Lab Arguments", "Arguments specific to Isaac Lab framework")

    isaac_lab_group.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    isaac_lab_group.add_argument(
        "--seed", type=int, default=None, help="Optional seed for the random number generator."
    )
    isaac_lab_group.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    # NOTE(alexmillane, 2025.07.25): Unlike base isaaclab, we enable pinocchio by default.
    isaac_lab_group.add_argument(
        "--disable_pinocchio",
        dest="enable_pinocchio",
        default=True,
        action="store_false",
        help="Disable Pinocchio.",
    )
    isaac_lab_group.add_argument("--mimic", action="store_true", default=False, help="Enable mimic environment.")


def add_external_environments_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add external environments specific command line arguments to the given parser."""
    external_environments_group = parser.add_argument_group(
        "External Environments Arguments", "Arguments specific to external environments"
    )
    external_environments_group.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Name of the external environment to run",
    )
