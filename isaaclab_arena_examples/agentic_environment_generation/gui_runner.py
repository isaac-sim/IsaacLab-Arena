# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI launcher for the ArenaEnvInitialGraphSpec live editor.

Usage:
    # Default — start with a prompt:
    python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

    # Open an existing spec:
    python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py \\
        --env_initial_graph_spec isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml

    # Custom port:
    python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py --port 8600

    # Custom output directory for generated YAML:
    python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py \\
        --out_dir isaaclab_arena_environments/agent_generated
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab_arena.agentic_environment_generation.spec_io import DEFAULT_AGENTIC_OUTPUT_DIR
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SIMAPP_SOCKET_ENV,
    SimAppError,
    spawn_simapp_process,
    stop_simapp_process,
    wait_for_simapp_socket,
)

_REVIEW_GUI_DIR = Path(__file__).resolve().parent / "review_gui"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env_initial_graph_spec",
        type=Path,
        default=None,
        help="Optional ArenaEnvInitialGraphSpec YAML to open in the editor.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_AGENTIC_OUTPUT_DIR,
        help=(
            "Directory for generated initial/linked spec YAML files (default:"
            " isaaclab_arena_environments/agent_generated)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501).",
    )
    args = parser.parse_args()
    serve_live_editor(args.env_initial_graph_spec, out_dir=args.out_dir, port=args.port)


def serve_live_editor(yaml_path: Path | None, *, out_dir: Path = DEFAULT_AGENTIC_OUTPUT_DIR, port: int = 8501) -> None:
    """Start the SimApp server, spawn Streamlit, and supervise both until exit."""
    app_path = _REVIEW_GUI_DIR / "streamlit_ui.py"
    assert app_path.exists(), f"Streamlit app not found at {app_path} — installation is incomplete."

    active_socket: Path | None = Path(tempfile.gettempdir()) / f"arena_review_simapp_{os.getpid()}.sock"
    simapp_proc = None
    try:
        print(f"[review_gui] booting Isaac Sim on {active_socket} …", file=sys.stderr)
        simapp_proc = spawn_simapp_process(str(active_socket))
        try:
            wait_for_simapp_socket(str(active_socket), simapp_proc)
        except SimAppError as exc:
            print(f"[review_gui] SimApp unavailable — continuing without it: {exc}", file=sys.stderr)
            stop_simapp_process(simapp_proc, str(active_socket))
            simapp_proc = None
            active_socket = None
        else:
            print("[review_gui] SimApp ready.", file=sys.stderr)

        env = os.environ.copy()
        if active_socket is not None:
            env[SIMAPP_SOCKET_ENV] = str(active_socket)

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
            cmd.extend(["--env_initial_graph_spec", str(yaml_path.resolve())])
        cmd.extend(["--out_dir", str(out_dir.resolve())])

        print(f"[review_gui] launching Streamlit live editor: {' '.join(cmd)}", file=sys.stderr)
        try:
            subprocess.run(cmd, env=env, check=True)
        except FileNotFoundError as exc:
            raise SystemExit(
                "Streamlit is not installed. Inside the isaaclab_arena container run:\n"
                "  python -m pip install --user --ignore-installed streamlit streamlit-ace"
            ) from exc
        except KeyboardInterrupt:
            pass
    finally:
        if active_socket is not None:
            print("[review_gui] shutting down SimApp …", file=sys.stderr)
            stop_simapp_process(simapp_proc, str(active_socket))


if __name__ == "__main__":
    main()
