# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI launcher for the ArenaEnvInitialGraphSpec live editor.

Spawns Streamlit with :mod:`~isaaclab_arena_examples.agentic_environment_generation.review_gui.streamlit_ui`.

Usage:
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.gui_runner \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml

    # Prompt-only (empty editor until you generate or paste YAML):
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.gui_runner

    # Prompt-only (empty editor until you generate or paste YAML):
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.gui_runner

    # Custom port:
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.gui_runner \\
        --yaml <path> --port 8600
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REVIEW_GUI_DIR = Path(__file__).resolve().parent / "review_gui"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Optional ArenaEnvInitialGraphSpec YAML to open in the editor.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501).",
    )
    args = parser.parse_args()
    serve_live_editor(args.yaml, port=args.port)


def serve_live_editor(yaml_path: Path | None, port: int = 8501) -> None:
    """Spawn ``streamlit run streamlit_ui.py`` and wait."""
    app_path = _REVIEW_GUI_DIR / "streamlit_ui.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path} — installation is incomplete.")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
        "--",
    ]
    if yaml_path is not None:
        cmd.extend(["--yaml", str(yaml_path.resolve())])

    print(f"[review_gui] launching Streamlit live editor: {' '.join(cmd)}", file=sys.stderr)
    try:
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed. Inside the isaaclab_arena container run:\n"
            "  python -m pip install --user --ignore-installed streamlit streamlit-ace"
        ) from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
