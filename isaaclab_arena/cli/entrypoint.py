# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab CLI extension for Arena workflows."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch an Arena command registered with the Isaac Lab CLI."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Run Isaac Lab Arena workflows.", prog="isaaclab arena")
    parser.add_argument("command", choices=("evaluate",), help="Arena workflow to run.")
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
    args = parser.parse_args(argv[:1])

    if args.command == "evaluate":
        from isaaclab_arena.evaluation.experiment_runner import main as run_experiment

        run_experiment(argv[1:])
        return 0
    raise AssertionError(f"Unsupported Arena command: {args.command}")
