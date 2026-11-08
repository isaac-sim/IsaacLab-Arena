# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the persistent SimulationApp backend used by ``scene_inspector.py``.

Usage::

    isaaclab.sh -p -m isaaclab_arena.agentic_tools.scene_inspector_backend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab_arena.agentic_tools.inspection.backend_process import select_socket_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        type=Path,
        default=select_socket_path(),
        help="Unix socket path shared with scene_inspector.py.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    from isaaclab_arena.agentic_tools.inspection.server import serve

    return serve(args.socket)


if __name__ == "__main__":
    sys.exit(main())
