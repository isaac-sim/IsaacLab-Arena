# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Long-lived ``SimulationApp`` host process for the live review editor.

Boots Kit's ``SimulationApp`` once on *its own* main thread and serves
spec-validation requests over a newline-delimited JSON-RPC pipe on
stdin/stdout. The parent (``streamlit_ui.py`` running inside Streamlit) spawns
exactly one of these and reuses it for the entire server lifetime via
:class:`isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar_client.SimAppSidecar`.

Why a sidecar and not an in-process ``SimulationApp``:

* ``signal.signal`` only works in the main thread. Streamlit's
  ``ScriptRunner`` runs the script in a worker thread, so SimApp's signal
  setup raises ``ValueError("signal only works in main thread …")``.
* Registry imports transitively load ``pxr``, which must happen only after
  ``SimulationApp`` starts. A dedicated process keeps that ordering safe.

Protocol (newline-delimited JSON over stdin/stdout):

  Ready handshake (sent by sidecar on boot before reading any request):
    {"ready": true}                          # SimApp boot succeeded
    {"ready": false, "error": "..."}         # boot failed; sidecar exits

  Requests:
    {"cmd": "ping"}
      → {"ok": true}

    {"cmd": "validate_spec", "yaml_text": "..."}
      → {"ok": true, "spec_dict": {...}}
        (full :class:`ArenaEnvInitialGraphSpec` validation including registry
         lookups — runs in the sidecar where registries are already warm)

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

import argparse
import contextlib
import json
import os
import signal
import sys
import traceback
import yaml
from typing import Any

# ---------------------------------------------------------------------------
# stdout multiplexing setup — run BEFORE importing anything that might
# touch Kit or print to stdout, otherwise Kit's chatter pollutes the JSON
# channel and the parent crashes on the first bad json.loads.
# ---------------------------------------------------------------------------

_JSON_FD = os.dup(1)  # save real stdout for our JSON channel
os.dup2(2, 1)
sys.stdout = sys.stderr


def _send(payload: dict[str, Any]) -> None:
    """Write one JSON line to the parent on the saved stdout fd."""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    os.write(_JSON_FD, data)


def _launch_simulation_app():
    """Boot Isaac Sim's ``SimulationApp`` for registry-backed validation."""
    try:
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher  # noqa: PLC0415

        sim_args = argparse.Namespace(headless=True, enable_cameras=False, hide_ui=True, livestream=-1)
        return get_app_launcher(sim_args).app
    except Exception as exc:
        print(f"[simapp_sidecar] SimulationApp launch failed: {exc}", file=sys.stderr)
        return None


def _install_signal_handlers() -> None:
    def _exit(signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


def _serve() -> int:
    """Boot SimApp, hand-shake with the parent, then service validation requests."""
    _install_signal_handlers()

    app = _launch_simulation_app()
    if app is None:
        _send({"ready": False, "error": "SimulationApp launch returned None"})
        return 1

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

            if cmd == "validate_spec":
                _send(_handle_validate_spec(req, ArenaEnvInitialGraphSpec))
                continue

            _send({"ok": False, "error": f"unknown cmd: {cmd!r}"})

        return 0
    finally:
        with contextlib.suppress(Exception):
            app.close()


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


def main() -> int:
    return _serve()


if __name__ == "__main__":
    sys.exit(main())
