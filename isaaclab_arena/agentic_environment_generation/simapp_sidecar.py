# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Long-lived ``SimulationApp`` host process for the live review editor.

Boots Kit's ``SimulationApp`` once on *its own* main thread and serves
thumbnail-render requests over a newline-delimited JSON-RPC pipe on
stdin/stdout. The parent (``review_app.py`` running inside Streamlit) spawns
exactly one of these and reuses it for the entire server lifetime via
:class:`isaaclab_arena.agentic_environment_generation.review_graph.SimAppSidecar`.

Why a sidecar and not an in-process ``SimulationApp``:

* ``signal.signal`` only works in the main thread. Streamlit's
  ``ScriptRunner`` runs the script in a worker thread, so SimApp's signal
  setup raises ``ValueError("signal only works in main thread …")``.
* ``omni.usd.UsdContext`` is process-singleton AND can't tolerate cross-
  thread driving from Streamlit reruns — driving it from worker threads
  triggers ``[Error] [omni.usd] UsdContext busy`` and the open_stage call
  fails. A dedicated process with serialized request handling avoids both.

Protocol (newline-delimited JSON over stdin/stdout):

  Ready handshake (sent by sidecar on boot before reading any request):
    {"ready": true}                          # SimApp boot succeeded
    {"ready": false, "error": "..."}         # boot failed; sidecar exits

  Requests:
    {"cmd": "ping"}
      → {"ok": true}

    {"cmd": "render_spec", "yaml_text": "..."}
      → {"ok": true, "paths": {"node_id": "/abs/path/to.png", ...},
                       "errors": [{"node_id": "...", "error": "..."}]}
        (paths are absolute filesystem paths on the disk cache. The PNGs
         themselves stay on disk — the parent reads them itself.)

    {"cmd": "shutdown"}
      → {"ok": true}   # sidecar exits cleanly after replying

  Parent EOF on stdin (parent process died) triggers the same graceful
  shutdown as the explicit "shutdown" cmd.

stdout multiplexing:

Kit writes a lot to stdout (warnings, replicator startup, etc.) and that
would corrupt the JSON channel the parent reads. We dup the original
stdout fd before touching Kit, then redirect Kit's stdout to stderr —
JSON replies go out through the saved fd; everything else from Kit
appears on the user's terminal via inherited stderr.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import traceback
import yaml
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# stdout multiplexing setup — run BEFORE importing anything that might
# touch Kit or print to stdout, otherwise Kit's chatter pollutes the JSON
# channel and the parent crashes on the first bad json.loads.
# ---------------------------------------------------------------------------

_JSON_FD = os.dup(1)  # save real stdout for our JSON channel
# Redirect fd 1 to stderr at the OS level so writes from C-extensions (Kit)
# end up on the terminal instead of the JSON pipe.
os.dup2(2, 1)
# Mirror at the Python level so `print()` and python-level stdout writes
# also go to the terminal. ``sys.stderr`` already points to the inherited
# parent stderr.
sys.stdout = sys.stderr


def _send(payload: dict[str, Any]) -> None:
    """Write one JSON line to the parent on the saved stdout fd."""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    os.write(_JSON_FD, data)


def _install_signal_handlers() -> None:
    """Translate SIGINT / SIGTERM to a clean ``SystemExit``.

    Lets the ``finally`` block in :func:`_serve` close ``SimulationApp``
    even when the parent kills us with a signal rather than the explicit
    shutdown command.
    """

    def _exit(signum, _frame):
        # SystemExit is preferable to ``os._exit`` here: it propagates
        # through the for-loop in ``_serve`` and reaches our finally.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


def _serve() -> int:
    """Boot SimApp, hand-shake with the parent, then service render requests.

    Returns the process exit code so :func:`main` can propagate it.
    """
    _install_signal_handlers()

    try:
        # Importing review_graph is light (no Kit yet); the SimApp launch is
        # what costs ~30s.
        from isaaclab_arena.agentic_environment_generation.review_graph import _launch_simulation_app  # noqa: PLC0415
    except Exception as exc:
        _send({"ready": False, "error": f"import failed: {exc}", "traceback": traceback.format_exc()})
        return 1

    app = _launch_simulation_app()
    if app is None:
        _send({"ready": False, "error": "SimulationApp launch returned None"})
        return 1

    # Post-Kit imports — these touch pxr transitively and MUST come after
    # SimApp boot (same reason ``_resolve_node_usd_paths`` is lazy).
    from isaaclab_arena.agentic_environment_generation.review_graph import _render_thumbnails_with_app  # noqa: PLC0415
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec  # noqa: PLC0415

    _send({"ready": True})

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                _send({"ok": False, "error": f"bad json: {exc}"})
                continue

            cmd = req.get("cmd")
            if cmd == "shutdown":
                _send({"ok": True})
                return 0

            if cmd == "ping":
                _send({"ok": True})
                continue

            if cmd == "render_spec":
                _send(_handle_render_spec(app, req, _render_thumbnails_with_app, ArenaEnvInitialGraphSpec))
                continue

            _send({"ok": False, "error": f"unknown cmd: {cmd!r}"})

        # ``for raw_line in sys.stdin`` exits when the parent's write end of
        # the pipe is closed — i.e. the parent died. Treat that the same as
        # a polite shutdown.
        return 0
    finally:
        with contextlib.suppress(Exception):
            app.close()


def _handle_render_spec(
    app,
    req: dict[str, Any],
    render_fn,
    spec_cls,
) -> dict[str, Any]:
    """Parse the spec, run thumbnail rendering, marshal the response.

    Kept separate from :func:`_serve` so each request gets its own try/except
    boundary — one bad spec shouldn't tear down the sidecar.
    """
    yaml_text = req.get("yaml_text")
    if not isinstance(yaml_text, str):
        return {"ok": False, "error": "render_spec requires string 'yaml_text'"}

    try:
        spec = spec_cls.from_dict(yaml.safe_load(yaml_text))
    except Exception as exc:
        return {"ok": False, "error": f"spec parse failed: {exc}", "traceback": traceback.format_exc()}

    try:
        # ``_render_thumbnails_with_app`` was refactored to return
        # ``dict[node_id, Path]`` (paths on the disk cache); the parent reads
        # the PNG bytes itself to keep IPC small.
        paths: dict[str, Path] = render_fn(app, spec)
    except Exception as exc:
        return {"ok": False, "error": f"render failed: {exc}", "traceback": traceback.format_exc()}

    return {
        "ok": True,
        "paths": {node_id: str(p) for node_id, p in paths.items()},
        "errors": [],
    }


def main() -> int:
    return _serve()


if __name__ == "__main__":
    sys.exit(main())
