# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Long-lived SimulationApp server for the review GUI (Unix socket JSON-RPC)."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import socket
import sys
import traceback
import yaml
from pathlib import Path
from typing import Any, TextIO

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.boot import launch_simulation_app


def _install_signal_handlers() -> None:
    def _exit(signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


def _write_response(writer: TextIO, payload: dict[str, Any]) -> None:
    writer.write(json.dumps(payload) + "\n")
    writer.flush()


def _serve_connection(
    reader: TextIO,
    writer: TextIO,
    *,
    app,
    render_fn,
    spec_cls,
) -> bool:
    """Handle JSON-RPC requests until disconnect; return True after ``shutdown``."""
    for raw_line in reader:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_response(writer, {"ok": False, "error": f"bad json: {exc}"})
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            _write_response(writer, {"ok": True})
            return True

        if cmd == "ping":
            _write_response(writer, {"ok": True})
            continue

        if cmd == "render_spec":
            _write_response(writer, _handle_render_spec(app, req, render_fn, spec_cls))
            continue

        if cmd == "run_sim_preview":
            _write_response(writer, _handle_run_sim_preview(app, req))
            continue

        _write_response(writer, {"ok": False, "error": f"unknown cmd: {cmd!r}"})

    return False


def _serve_socket(socket_path: str) -> int:
    """Boot SimApp, bind ``socket_path``, and service requests sequentially."""
    _install_signal_handlers()

    app = launch_simulation_app()
    if app is None:
        print("[simapp] SimulationApp launch returned None", file=sys.stderr)
        return 1

    # Kit/Omniverse and pxr (via asset_usd) must load only after SimulationApp starts.
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.thumbnail_capture import (  # noqa: PLC0415
        render_thumbnails_with_app,
    )

    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(5)
    print(f"[simapp] listening on {path}", file=sys.stderr)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                reader = conn.makefile("r", encoding="utf-8", newline="\n")
                writer = conn.makefile("w", encoding="utf-8", newline="\n")
                try:
                    should_exit = _serve_connection(
                        reader,
                        writer,
                        app=app,
                        render_fn=render_thumbnails_with_app,
                        spec_cls=ArenaEnvInitialGraphSpec,
                    )
                finally:
                    reader.close()
                    writer.close()
                if should_exit:
                    break
    finally:
        server.close()
        path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            app.close()

    return 0


def _handle_render_spec(
    app,
    req: dict[str, Any],
    render_fn,
    spec_cls,
) -> dict[str, Any]:
    """Parse the spec, run thumbnail rendering, marshal the response."""
    yaml_text = req.get("yaml_text")
    if not isinstance(yaml_text, str):
        return {"ok": False, "error": "render_spec requires string 'yaml_text'"}

    try:
        spec = spec_cls.from_dict(yaml.safe_load(yaml_text))
    except Exception as exc:
        return {"ok": False, "error": f"spec parse failed: {exc}", "traceback": traceback.format_exc()}

    try:
        paths, aabb_dimensions_m = render_fn(app, spec)
    except Exception as exc:
        return {"ok": False, "error": f"render failed: {exc}", "traceback": traceback.format_exc()}

    return {
        "ok": True,
        "paths": {node_id: str(p) for node_id, p in paths.items()},
        "aabb_dimensions_m": {node_id: list(dims) for node_id, dims in aabb_dimensions_m.items()},
    }


def _handle_run_sim_preview(app, req: dict[str, Any]) -> dict[str, Any]:
    """Handle run_sim_preview JSON-RPC."""
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.sim_preview import (  # noqa: PLC0415
        parse_sim_preview_params,
        run_sim_preview,
    )

    yaml_text = req.get("yaml_text")
    if not isinstance(yaml_text, str):
        return {"ok": False, "error": "run_sim_preview requires string 'yaml_text'"}

    try:
        num_envs, num_steps, env_spacing = parse_sim_preview_params(req)
    except (TypeError, ValueError, AssertionError) as exc:
        return {"ok": False, "error": f"invalid sim preview params: {exc}"}

    try:
        return run_sim_preview(
            app,
            yaml_text,
            num_envs=num_envs,
            num_steps=num_steps,
            env_spacing=env_spacing,
        )
    except Exception as exc:
        return {"ok": False, "error": f"sim preview failed: {exc}", "traceback": traceback.format_exc()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        required=True,
        help="Unix domain socket path for newline-delimited JSON-RPC requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return _serve_socket(args.socket)


if __name__ == "__main__":
    sys.exit(main())
