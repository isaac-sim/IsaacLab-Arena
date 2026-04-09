# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Warm the Kit shader cache by launching and closing a SimulationApp.

On CI (fresh container, no cached shaders) the first SimulationApp startup
compiles hundreds of RTX/rendering shaders, taking 60-180 s on constrained
runners.  Running this script once before tests ensures every subsequent
SimulationApp — persistent or subprocess-spawned — gets a warm-cache start.

Usage:
    /isaac-sim/python.sh scripts/warm_shader_cache.py
"""

import argparse
import time

from isaaclab.app import AppLauncher


def main():
    t0 = time.monotonic()
    args = argparse.Namespace(headless=True, enable_cameras=True, visualizer=[])
    launcher = AppLauncher(args)
    elapsed_launch = time.monotonic() - t0
    print(f"[warm_shader_cache] AppLauncher ready in {elapsed_launch:.1f}s")

    launcher.app.close()
    elapsed_total = time.monotonic() - t0
    print(f"[warm_shader_cache] Done in {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
