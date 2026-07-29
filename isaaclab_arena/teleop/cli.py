# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI normalization shared by Arena teleoperation scripts."""

import argparse


def enable_openxr_teleop_from_cli(args_cli: argparse.Namespace) -> None:
    """Select OpenXR teleoperation when ``--xr`` is the only XR option provided."""
    teleop_device = getattr(args_cli, "teleop_device", None)
    if getattr(args_cli, "xr", False) and teleop_device is None:
        args_cli.teleop_device = "openxr"
        teleop_device = args_cli.teleop_device
    if isinstance(teleop_device, str) and teleop_device.lower() == "openxr":
        args_cli.xr = True
