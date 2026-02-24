#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open a USD stage in Isaac Sim and keep the app alive for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Open a USD stage in Isaac Sim for manual inspection.")
parser.add_argument("--usd_path", type=Path, required=True, help="USD or USDA file to open in Isaac Sim.")
parser.add_argument(
    "--wait_frames",
    type=int,
    default=4,
    help="Number of app update frames after opening the stage before reporting success.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import omni.usd

    usd_path = args_cli.usd_path.expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD stage does not exist: {usd_path}")

    usd_context = omni.usd.get_context()
    opened = usd_context.open_stage(str(usd_path))
    if not opened:
        raise RuntimeError(f"Failed to open stage: {usd_path}")

    for _ in range(max(args_cli.wait_frames, 0)):
        simulation_app.update()

    stage = usd_context.get_stage()
    default_prim = stage.GetDefaultPrim().GetPath().pathString if stage and stage.GetDefaultPrim() else "<none>"
    print(f"[open_usd_stage] opened={usd_path}")
    print(f"[open_usd_stage] default_prim={default_prim}")

    if not args_cli.headless:
        print("[open_usd_stage] GUI mode active. Inspect the stage and close Isaac Sim when done.")
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
