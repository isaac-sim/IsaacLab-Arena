# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimulationApp boot helpers for the review GUI SimApp subprocess."""

from __future__ import annotations

import argparse
import sys

from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher


def launch_args(*, enable_visualizer: bool = True) -> argparse.Namespace:
    """AppLauncher args for the review GUI SimApp.

    Args:
        enable_visualizer: If True (default), boot with Kit UI for viewport capture. If False,
            boot headless with cameras (CI-friendly).
    """
    if not enable_visualizer:
        # Match Arena's camera-enabled CI path: headless host, offscreen cameras still available.
        return argparse.Namespace(headless=True, enable_cameras=True, livestream=-1)
    return argparse.Namespace(visualizer=["kit"], enable_cameras=True, livestream=-1)


def launch_simulation_app(*, enable_visualizer: bool = True):
    """Boot Isaac Sim's ``SimulationApp``, or ``None`` on failure.

    Args:
        enable_visualizer: Forwarded to :func:`launch_args`. Defaults to Kit UI.
    """
    try:
        return get_app_launcher(launch_args(enable_visualizer=enable_visualizer)).app
    except Exception as exc:
        print(f"[simapp] SimulationApp launch failed: {exc}", file=sys.stderr)
        return None
