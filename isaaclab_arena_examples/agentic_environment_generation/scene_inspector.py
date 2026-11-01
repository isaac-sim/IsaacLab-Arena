# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect authored USD scene data without starting SimulationApp.

Usage::

    python isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py \
        prim-tree --usd /path/to/scene.usd --output prim_tree.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _json_stdout(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _dump_prim_tree(args: argparse.Namespace) -> int:
    from isaaclab_arena.utils.usd.prim_tree import format_usd_prim_tree, load_usd_prim_tree

    started_at = time.monotonic()
    records = load_usd_prim_tree(args.usd)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_usd_prim_tree(records) + "\n", encoding="utf-8")
    _json_stdout({
        "command": "prim-tree",
        "status": "completed",
        "inputs": {"usd": args.usd},
        "output": str(output_path),
        "record_count": len(records),
        "timings_s": {"total": time.monotonic() - started_at},
    })
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prim_tree = commands.add_parser("prim-tree", help="Dump the USD physics/collision prim tree.")
    prim_tree.add_argument("--usd", required=True, help="Local path or supported remote URL of the USD scene.")
    prim_tree.add_argument("--output", type=Path, required=True, help="Destination text file.")
    prim_tree.set_defaults(handler=_dump_prim_tree)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.handler(args))
    except (AssertionError, OSError, RuntimeError, ValueError) as exc:
        _json_stdout({
            "command": args.command,
            "status": "failed",
            "error_code": type(exc).__name__,
            "error": str(exc),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
