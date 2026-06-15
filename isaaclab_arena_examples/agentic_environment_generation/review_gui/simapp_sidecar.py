# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Long-lived ``SimulationApp`` host process for the live review editor.

Boots Kit's ``SimulationApp`` once on *its own* main thread and serves
validation and thumbnail-render requests over a newline-delimited JSON-RPC
pipe on stdin/stdout. The parent (``streamlit_ui.py`` running inside
Streamlit) spawns exactly one of these and reuses it for the entire server
lifetime via
:class:`isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar_client.SimAppSidecar`.

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

    {"cmd": "validate_spec", "yaml_text": "..."}
      → {"ok": true, "spec_dict": {...}}
        (full :class:`ArenaEnvInitialGraphSpec` validation including registry
         lookups — runs in the sidecar where registries are already warm)

    {"cmd": "render_spec", "yaml_text": "..."}
      → {"ok": true, "paths": {"node_id": "/abs/path/to.png", ...},
                       "errors": [{"node_id": "...", "error": "..."}]}
        (paths are absolute filesystem paths on the disk cache. The PNGs
         themselves stay on disk — the parent reads them itself.)

    {"cmd": "build_catalogues"}
      → {"ok": true, "asset_catalogue": {...}, "relation_catalogue": {...},
                       "task_catalogue": {...}}
        (registry vocabulary for :meth:`EnvironmentGenerationAgent.fetch_intent_from_prompt`)

    {"cmd": "compile_intent", "intent_dict": {...}}
      → {"ok": true, "spec_dict": {...}, "has_resolution_errors": bool,
                       "trace": [{"stage": "...", "query": "...", ...}]}
        (validates :class:`EnvironmentIntentSpec` and compiles to initial graph spec)

    {"cmd": "run_sim_preview", "yaml_text": "..."}
      → {"ok": true, "first_frame": "/abs/first.png", "last_frame": "/abs/last.png",
                       "num_envs": 16, "env_spacing": 1.5, "num_steps": 50}
        (link → to_arena_env → relation solver → 50 zero-action steps; overview captures)

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

_JSON_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def _send(payload: dict[str, Any]) -> None:
    """Write one JSON line to the parent on the saved stdout fd."""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    os.write(_JSON_FD, data)


def _install_signal_handlers() -> None:
    def _exit(signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


def _serve() -> int:
    """Boot SimApp, hand-shake with the parent, then service requests."""
    _install_signal_handlers()

    try:
        from isaaclab_arena_examples.agentic_environment_generation.review_gui.thumbnail_render import (  # noqa: PLC0415
            _launch_simulation_app,
        )
    except Exception as exc:
        _send({"ready": False, "error": f"import failed: {exc}", "traceback": traceback.format_exc()})
        return 1

    app = _launch_simulation_app()
    if app is None:
        _send({"ready": False, "error": "SimulationApp launch returned None"})
        return 1

    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec  # noqa: PLC0415
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.thumbnail_render import (  # noqa: PLC0415
        _render_thumbnails_with_app,
    )

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

            if cmd == "render_spec":
                _send(_handle_render_spec(app, req, _render_thumbnails_with_app, ArenaEnvInitialGraphSpec))
                continue

            if cmd == "build_catalogues":
                _send(_handle_build_catalogues())
                continue

            if cmd == "compile_intent":
                _send(_handle_compile_intent(req))
                continue

            if cmd == "run_sim_preview":
                _send(_handle_run_sim_preview(app, req))
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


def _handle_run_sim_preview(app, req: dict[str, Any]) -> dict[str, Any]:
    """Build linked env, solve relations, roll out zero actions, capture overview frames."""
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.sim_preview import (  # noqa: PLC0415
        run_sim_preview,
    )

    yaml_text = req.get("yaml_text")
    if not isinstance(yaml_text, str):
        return {"ok": False, "error": "run_sim_preview requires string 'yaml_text'"}

    try:
        return run_sim_preview(app, yaml_text)
    except Exception as exc:
        return {"ok": False, "error": f"sim preview failed: {exc}", "traceback": traceback.format_exc()}


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


def main() -> int:
    return _serve()


if __name__ == "__main__":
    sys.exit(main())
