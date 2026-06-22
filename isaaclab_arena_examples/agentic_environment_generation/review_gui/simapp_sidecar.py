# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Long-lived ``SimulationApp`` host process for the live review editor.

Boots Kit's ``SimulationApp`` once (with ``--viz kit``) on *its own* main thread and serves
validation and thumbnail-render requests over a newline-delimited JSON-RPC protocol on a
Unix domain socket. ``gui_runner`` spawns exactly one of these per review session and
exports the socket path to Streamlit via ``ARENA_REVIEW_SIDECAR_SOCKET``.

Why a sidecar and not an in-process ``SimulationApp``:

* ``signal.signal`` only works in the main thread. Streamlit's
  ``ScriptRunner`` runs the script in a worker thread, so SimApp's signal
  setup raises ``ValueError("signal only works in main thread …")``.
* ``omni.usd.UsdContext`` is process-singleton AND can't tolerate cross-
  thread driving from Streamlit reruns — driving it from worker threads
  triggers ``[Error] [omni.usd] UsdContext busy`` and the open_stage call
  fails. A dedicated process with serialized request handling avoids both.

Protocol (newline-delimited JSON over a Unix domain socket):

  Requests:
    {"cmd": "ping"}
      → {"ok": true}

    {"cmd": "validate_spec", "yaml_text": "..."}
      → {"ok": true, "spec_dict": {...}}

    {"cmd": "render_spec", "yaml_text": "..."}
      → {"ok": true, "paths": {"node_id": "/abs/path/to.png", ...},
                       "errors": [{"node_id": "...", "error": "..."}]}

    {"cmd": "build_catalogues"}
      → {"ok": true, "asset_catalogue": {...}, "relation_catalogue": {...},
                       "task_catalogue": {...}}

    {"cmd": "compile_intent", "intent_dict": {...}}
      → {"ok": true, "spec_dict": {...}, "has_resolution_errors": bool,
                       "trace": [{"stage": "...", "query": "...", ...}]}

    {"cmd": "shutdown"}
      → {"ok": true}   # sidecar exits cleanly after replying

stdout multiplexing:

Kit writes a lot to stdout (warnings, replicator startup, etc.) and that
would corrupt any JSON channel on stdout. We redirect Kit's stdout to
stderr so diagnostics appear on the user's terminal via inherited stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import sys
import traceback
import yaml
from pathlib import Path
from typing import Any, TextIO

os.dup2(2, 1)
sys.stdout = sys.stderr


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
    """Handle JSON-RPC requests on one connected client until disconnect.

    Returns:
        True when the client sent ``shutdown`` and the sidecar should exit.
    """
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

        if cmd == "validate_spec":
            _write_response(writer, _handle_validate_spec(req, spec_cls))
            continue

        if cmd == "render_spec":
            _write_response(writer, _handle_render_spec(app, req, render_fn, spec_cls))
            continue

        if cmd == "build_catalogues":
            _write_response(writer, _handle_build_catalogues())
            continue

        if cmd == "compile_intent":
            _write_response(writer, _handle_compile_intent(req))
            continue

        _write_response(writer, {"ok": False, "error": f"unknown cmd: {cmd!r}"})

    return False


def _serve_socket(socket_path: str) -> int:
    """Boot SimApp, bind ``socket_path``, and service requests sequentially."""
    _install_signal_handlers()

    try:
        from isaaclab_arena_examples.agentic_environment_generation.review_gui.thumbnail_render import (  # noqa: PLC0415
            _launch_simulation_app,
        )
    except Exception as exc:
        print(f"[simapp_sidecar] import failed: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1

    app = _launch_simulation_app()
    if app is None:
        print("[simapp_sidecar] SimulationApp launch returned None", file=sys.stderr)
        return 1

    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec  # noqa: PLC0415
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.thumbnail_render import (  # noqa: PLC0415
        _render_thumbnails_with_app,
    )

    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(5)
    print(f"[simapp_sidecar] listening on {path}", file=sys.stderr)

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
                        render_fn=_render_thumbnails_with_app,
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


def _handle_validate_spec(req: dict[str, Any], spec_cls) -> dict[str, Any]:
    """Parse and fully validate YAML as an initial graph spec."""
    yaml_text = req.get("yaml_text")
    if not isinstance(yaml_text, str):
        return {"ok": False, "error": "validate_spec requires string 'yaml_text'"}

    try:
        raw = yaml.safe_load(yaml_text)
    except Exception as exc:
        return {"ok": False, "error": f"yaml parse failed: {exc}", "traceback": traceback.format_exc()}

    if raw is None:
        return {"ok": False, "error": "YAML is empty"}
    if not isinstance(raw, dict):
        return {"ok": False, "error": f"expected mapping, got {type(raw).__name__}"}

    try:
        spec = spec_cls.model_validate(raw)
    except Exception as exc:
        return {"ok": False, "error": f"spec validation failed: {exc}", "traceback": traceback.format_exc()}

    return {"ok": True, "spec_dict": spec.to_dict()}


def _handle_build_catalogues() -> dict[str, Any]:
    """Return asset/relation/task catalogues for the env-generation agent."""
    from dataclasses import asdict  # noqa: PLC0415

    from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (  # noqa: PLC0415
        build_asset_catalogue,
        build_relation_catalogue,
        build_task_catalogue,
    )

    try:
        asset_catalogue = build_asset_catalogue()
        relation_catalogue = build_relation_catalogue()
        task_catalogue = build_task_catalogue()
    except Exception as exc:
        return {"ok": False, "error": f"catalogue build failed: {exc}", "traceback": traceback.format_exc()}

    return {
        "ok": True,
        "asset_catalogue": asdict(asset_catalogue),
        "relation_catalogue": {
            "relations": [asdict(entry) for entry in relation_catalogue.relations],
        },
        "task_catalogue": {
            "tasks": [asdict(entry) for entry in task_catalogue.tasks],
        },
    }


def _handle_compile_intent(req: dict[str, Any]) -> dict[str, Any]:
    """Validate an EnvironmentIntentSpec and compile it to an initial graph spec."""
    from dataclasses import asdict  # noqa: PLC0415

    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import (  # noqa: PLC0415
        EnvironmentIntentSpec,
    )
    from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler  # noqa: PLC0415

    intent_dict = req.get("intent_dict")
    if not isinstance(intent_dict, dict):
        return {"ok": False, "error": "compile_intent requires mapping 'intent_dict'"}

    try:
        intent = EnvironmentIntentSpec.model_validate(intent_dict)
        compiler = IntentCompiler()
        spec = compiler.compile(intent)
    except Exception as exc:
        return {"ok": False, "error": f"intent compile failed: {exc}", "traceback": traceback.format_exc()}

    return {
        "ok": True,
        "spec_dict": spec.to_dict(),
        "has_resolution_errors": compiler.has_resolution_errors,
        "trace": [asdict(event) for event in compiler.trace],
        "reasoning": intent.reasoning,
    }


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
        paths: dict[str, Path] = render_fn(app, spec)
    except Exception as exc:
        return {"ok": False, "error": f"render failed: {exc}", "traceback": traceback.format_exc()}

    return {
        "ok": True,
        "paths": {node_id: str(p) for node_id, p in paths.items()},
        "errors": [],
    }


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
