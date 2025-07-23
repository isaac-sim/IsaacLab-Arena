# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import argparse

from isaaclab.app import AppLauncher


def get_isaac_arena_cli_parser() -> argparse.ArgumentParser:
    """Get a complete argument parser with both Isaac Lab and Isaac Arena arguments."""
    parser = argparse.ArgumentParser(description="Isaac Arena CLI parser.")
    add_isaac_lab_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def add_isaac_lab_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add Isaac Lab specific command line arguments to the given parser."""

    isaac_lab_group = parser.add_argument_group("Isaac Lab Arguments", "Arguments specific to Isaac Lab framework")

    isaac_lab_group.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    isaac_lab_group.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    isaac_lab_group.add_argument("--task", type=str, default=None, help="Name of the task.")
    isaac_lab_group.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Enable Pinocchio.",
    )
