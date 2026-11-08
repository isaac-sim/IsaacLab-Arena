# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sequential, main-thread inspection service over local JSON lines."""

from __future__ import annotations

import contextlib
import signal
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

from isaaclab_arena.agentic_tools.inspection.protocol import (
    encode_record,
    read_record,
    validate_request,
    validate_result,
)

REQUEST_READ_TIMEOUT_S = 5.0
RESPONSE_WRITE_TIMEOUT_S = 5.0


def _handle_render_spec(app, req: dict[str, Any], render_fn, spec_cls) -> dict[str, Any]:
    """Validate a graph and return completed snapshot artifacts."""
    spec = spec_cls.from_dict(req["spec"])
    options = req.get("options", {})
    paths, dimensions, warnings = render_fn(
        app,
        spec,
        background_panorama=options.get("background_panorama", False),
        ignore_cache=options.get("ignore_cache", False),
    )
    return {
        "artifacts": {key: {"path": str(path), "media_type": "image/png"} for key, path in paths.items()},
        "metadata": {"aabb_dimensions_m": {key: list(dims) for key, dims in dimensions.items()}},
        "warnings": warnings,
        "temporary_dir": None,
    }


def _dispatch(app, req: dict[str, Any], render_fn, spec_cls) -> dict[str, Any]:
    validate_request(req)
    if req["cmd"] in ("ping", "shutdown"):
        return {"ok": True}
    if req["cmd"] == "render_spec":
        result = _handle_render_spec(app, req, render_fn, spec_cls)
    else:
        from isaaclab_arena.agentic_tools.inspection.sim_preview import run_sim_preview

        spec = spec_cls.from_dict(req["spec"])
        result = run_sim_preview(app, spec, **req["options"])
    return {"ok": True, "result": validate_result(result)}


def _serve_connection(conn: socket.socket, *, app, render_fn, spec_cls, read_timeout_s=REQUEST_READ_TIMEOUT_S) -> bool:
    """Serve exactly one request; return whether a shutdown was accepted."""
    should_exit = False
    try:
        req = read_record(conn, timeout_s=read_timeout_s)
        response = _dispatch(app, req, render_fn, spec_cls)
        should_exit = req["cmd"] == "shutdown"
        record = encode_record(response)
    except Exception as exc:
        record = encode_record({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    try:
        conn.settimeout(RESPONSE_WRITE_TIMEOUT_S)
        conn.sendall(record)
    except OSError:
        # A disconnected or timed-out reader does not take down the backend.
        pass
    return should_exit


def _serve_requests(server: socket.socket, *, app, render_fn, spec_cls, read_timeout_s=REQUEST_READ_TIMEOUT_S) -> None:
    """Poll for requests and pump Kit on the main thread while idle."""
    server.settimeout(0.1)
    while True:
        try:
            conn, _ = server.accept()
        except TimeoutError:
            app.update()
            continue
        with conn:
            if _serve_connection(conn, app=app, render_fn=render_fn, spec_cls=spec_cls, read_timeout_s=read_timeout_s):
                return


def serve(socket_path: Path) -> int:
    """Bind an unused address, boot SimulationApp, and service requests until shutdown."""
    from isaaclab_arena.agentic_tools.inspection.boot import launch_simulation_app

    def _exit(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    app = None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        # bind() refuses occupied and stale addresses alike. Never unlink an address to probe it.
        server.bind(str(path))
        identity = path.stat().st_ino
        try:
            server.listen(16)
            with contextlib.redirect_stdout(sys.stderr):
                app = launch_simulation_app()
                if app is None:
                    return 1
                from isaaclab_arena.agentic_tools.inspection.thumbnail_capture import render_thumbnails_with_app
                from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

                print(f"[inspection] listening on {path}", file=sys.stderr)
                _serve_requests(server, app=app, render_fn=render_thumbnails_with_app, spec_cls=ArenaEnvGraphSpec)
        finally:
            if path.exists() and path.stat().st_ino == identity:
                path.unlink()
            if app is not None:
                with contextlib.suppress(Exception), contextlib.redirect_stdout(sys.stderr):
                    app.close()
    return 0
