# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimulationApp boot helpers for the review GUI SimApp subprocess."""

from __future__ import annotations

import argparse
import sys


def launch_args() -> argparse.Namespace:
    """AppLauncher args for the review GUI SimApp (Kit UI + viewport capture)."""
    return argparse.Namespace(visualizer=["kit"], enable_cameras=True, livestream=-1)


def launch_simulation_app():
    """Boot Isaac Sim's ``SimulationApp`` with the Kit visualizer, or ``None`` on failure."""
    try:
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher  # noqa: PLC0415

        return get_app_launcher(launch_args()).app
    except Exception as exc:
        print(f"[simapp] SimulationApp launch failed: {exc}", file=sys.stderr)
        return None
