# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live UnresolvedArenaEnvGraphSpec review tool — Streamlit editor + USD thumbnails.

The CLI is a thin launcher: it always boots the Streamlit app in
``review_app.py``. The static-HTML mode and the old ``--serve`` /
``--render-thumbnails`` switches were collapsed away — thumbnails are now
always rendered (cache-hit when possible, ``SimulationApp`` cache-miss
otherwise), and the result is shown inside the live editor.

The tool accepts an ``UnresolvedArenaEnvGraphSpec`` YAML as input. The user
edits the spec directly inside the Streamlit editor and the preview updates
in real time.

Three panels (dark dashboard style) inside the embedded view:
  * Top-left — graph diagram (mermaid.js, CDN-loaded) of the initial-state
    spatial constraints. Anchor nodes are highlighted; constraints without
    a reference (is_anchor / position_limits / at_pose / ...) are listed below
    the graph rather than rendered as self-loops.
  * Bottom-left — task table (index, kind, description, params).
  * Right — node card grid: type badge, asset name, and the per-node YAML
    stanza. The per-node thumbnail is a real USD viewport capture (cached
    on disk under ``.cache/llm_env_gen_thumbnails/`` and inlined as base64
    so the HTML stays self-contained).

Usage:
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.review_graph \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_unresolved.yaml

    # Custom port:
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.review_graph \\
        --yaml <path> --port 8600

Public API used by ``review_app.py``:
    * :func:`launch_simulation_app` — boots Kit's ``SimulationApp`` (headless +
      cameras). Returns ``None`` on failure so the app can degrade to
      placeholder thumbnails rather than crashing.
    * :func:`render_thumbnails_for_spec` — given a live ``SimulationApp`` and
      a parsed spec, returns ``{node_id: png_bytes}``. Cache-aware: existing
      PNGs under the disk cache are read directly; missing ones are rendered
      through the live app and written back to the cache for next time.
    * :func:`render_html_for_spec` — full HTML payload with the given
      thumbnails dict inlined. Pass an empty dict to fall back to placeholders.

Note on USD rendering:
    ``pxr.UsdAppUtils.FrameRecorder`` and the ``usdrecord`` CLI are NOT
    available inside the Isaac Sim container (Kit ships ``UsdAppUtils.py``
    but strips out ``libusd_usdAppUtils.so``, and ``usdrecord`` is omitted
    entirely). The Kit-equivalent path used here is:
    ``omni.usd`` to open the stage + ``omni.kit.viewport.utility`` to
    capture the active viewport. Kit transparently uses cached Nucleus
    thumbnails when opening ``omniverse://`` URIs, so we don't need a
    separate Nucleus-HTTPS probe path.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import html as html_lib
import json
import os
import re
import subprocess
import sys
import threading
import yaml
from pathlib import Path
from typing import Any

from isaaclab_arena.environments.arena_env_graph_spec import UnresolvedArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec, ArenaEnvGraphStateSpec

# Disk cache for rendered thumbnails. Keyed by sha1(usd_path) so identical
# USDs across envs reuse the same PNG. Survives across runs to avoid the
# ~30s SimulationApp boot when nothing changed.
_THUMBNAIL_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "llm_env_gen_thumbnails"
_THUMBNAIL_SIZE = 256


def main() -> None:
    """CLI entry point — argparse parses the user's flags, then we hand off
    to Streamlit. The actual interactive UI lives in ``review_app.py``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Path to an UnresolvedArenaEnvGraphSpec YAML file. The Streamlit app will open it for live editing.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501).",
    )
    args = parser.parse_args()
    _serve_live_editor(args.yaml, port=args.port)


def _serve_live_editor(yaml_path: Path, port: int = 8501) -> None:
    """Spawn ``streamlit run review_app.py -- --yaml <path>`` and wait.

    This is the only path through the CLI now — the old static-HTML and
    standalone ``--render-thumbnails`` flows were folded into the Streamlit
    app, which boots ``SimulationApp`` once via ``@st.cache_resource`` and
    keeps it alive for the lifetime of the server. We resolve
    ``review_app.py`` next to this file rather than going through ``-m`` so
    Streamlit picks the path up cleanly (``streamlit run`` doesn't accept
    module dotted-paths).
    """
    app_path = Path(__file__).with_name("review_app.py")
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
        # Skip the email prompt the first time Streamlit runs in a fresh
        # container — the live editor is a developer tool, not a hosted
        # service, and an interactive prompt would block automation.
        "--browser.gatherUsageStats",
        "false",
        # File watcher is a footgun here: Kit's ``SimulationApp`` boot is
        # tens of seconds; we don't want Streamlit silently rerunning the
        # script (and reissuing the cached_resource init) every time we
        # save a source file during development. The user can still hit "R"
        # in the browser to force a rerun if they want.
        "--server.fileWatcherType",
        "none",
        "--",
        "--yaml",
        str(yaml_path.resolve()),
    ]

    # Inherit env so the Streamlit subprocess sees PYTHONPATH / isaac-sim
    # site-packages exactly the same way we do.
    print(f"[review_graph] launching Streamlit live editor: {' '.join(cmd)}", file=sys.stderr)
    try:
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except FileNotFoundError as exc:
        # The plain ``pip install streamlit`` fails inside the isaaclab_arena
        # container because streamlit≥1.30 needs uvicorn>=0.30 but Kit ships
        # a bundled uvicorn==0.29 under a read-only /isaac-sim/extscache path.
        # ``--user --ignore-installed`` sidesteps the rollback by writing
        # everything to ~/.local (which is earlier on sys.path than extscache).
        raise SystemExit(
            "Streamlit is not installed. Inside the isaaclab_arena container run:\n"
            "  python -m pip install --user --ignore-installed streamlit streamlit-ace"
        ) from exc
    except KeyboardInterrupt:
        # Normal exit path — user hit Ctrl-C in the terminal.
        pass


# ---------------------------------------------------------------------------
# Public API consumed by review_app.py
# ---------------------------------------------------------------------------


def render_html_for_spec(spec: UnresolvedArenaEnvGraphSpec, thumbnails: dict[str, bytes] | None = None) -> str:
    """Render the review HTML for ``spec``, inlining the given thumbnails.

    Thin public alias of :func:`_render_html` so external entry points don't
    have to reach into a private name. Pass ``thumbnails=None`` (or omit) to
    fall back to placeholder thumbnails — useful when the sidecar is
    unavailable.
    """
    return _render_html(spec, thumbnails=thumbnails)


class SimAppSidecarError(RuntimeError):
    """Raised when the SimApp sidecar process can't fulfil a request.

    Distinct exception type so the Streamlit app can catch sidecar failures
    specifically (and e.g. clear its ``@st.cache_resource`` to force a
    re-spawn) without swallowing programming errors.
    """


class SimAppSidecar:
    """Long-lived Kit/SimApp host process exposed as a render service.

    See ``simapp_sidecar.py`` for the protocol. The instance is meant to be
    cached for the lifetime of the Streamlit server process via
    ``@st.cache_resource``; calling :meth:`render_spec` is safe across
    Streamlit reruns and across concurrent sessions (an internal
    ``threading.Lock`` serializes pipe access — Kit can only service one
    render at a time anyway).

    Lifecycle:

      * :meth:`start` spawns the subprocess and waits for the ``{"ready":
        true}`` handshake. Times out after ``boot_timeout_s`` if Kit boot
        hangs.
      * :meth:`render_spec` sends a ``render_spec`` request and reads the
        reply line. Reads paths back, materializes the PNG bytes from the
        shared filesystem cache, returns ``{node_id: bytes}``.
      * :meth:`close` sends ``shutdown`` and waits for the process to exit,
        terminating then killing if it doesn't.
      * On parent crash / SIGKILL, the sidecar reads EOF on stdin and exits
        on its own via the ``finally`` in ``simapp_sidecar._serve``.
    """

    # Subprocess.Popen would normally inherit the parent's stderr. Kit
    # writes a lot there, which is fine — we want users to see those logs.
    # The JSON channel travels through stdout instead; the sidecar redirects
    # Kit's stdout to stderr at the fd level before booting so the channel
    # stays clean.

    def __init__(self, *, boot_timeout_s: float = 180.0, shutdown_timeout_s: float = 10.0) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._boot_timeout_s = boot_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s

    # -- lifecycle --

    def start(self) -> None:
        """Spawn the sidecar process and wait for its ``{"ready": true}`` handshake.

        Raises :class:`SimAppSidecarError` if the boot fails (handshake says
        ``ready: false``, sidecar exits early, or boot takes longer than
        ``boot_timeout_s``).
        """
        if self._proc is not None and self._proc.poll() is None:
            return  # already running

        cmd = [
            sys.executable,
            "-m",
            "isaaclab_arena.agentic_environment_generation.simapp_sidecar",
        ]
        # ``start_new_session=False`` (default) leaves the child in the same
        # process group as the parent, so Ctrl-C in the launching terminal
        # also signals the sidecar. The sidecar installs SIGINT/SIGTERM
        # handlers that route to a clean SystemExit -> finally -> app.close().
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit so Kit logs flow to the user's terminal
            text=True,
            bufsize=1,  # line-buffered
            env=os.environ.copy(),
        )

        # Block until we hear the ready handshake. We don't use signal-based
        # timeout (only main thread can use ``signal.alarm``); a watchdog
        # thread is overkill here, so we just rely on the boot being fast
        # under normal conditions and let the user Ctrl-C if it really hangs.
        # In practice Kit either boots in ~30s or fails immediately.
        line = self._readline_or_die()
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            self._terminate()
            raise SimAppSidecarError(f"Sidecar emitted non-JSON handshake: {line!r}") from exc

        if not msg.get("ready"):
            self._terminate()
            raise SimAppSidecarError(
                f"Sidecar boot failed: {msg.get('error', 'unknown error')}\n{msg.get('traceback', '')}"
            )

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        """Send ``shutdown``, then terminate/kill if the process doesn't exit.

        Safe to call multiple times. Safe to call after the child has already
        died (e.g. via SIGINT propagated from the terminal). Quiet about
        common shutdown races so atexit doesn't spam the terminal.
        """
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            try:
                proc.wait(timeout=self._shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=5)

        with contextlib.suppress(Exception):
            if proc.stdin:
                proc.stdin.close()
        with contextlib.suppress(Exception):
            if proc.stdout:
                proc.stdout.close()

    # -- requests --

    def render_spec(self, spec: UnresolvedArenaEnvGraphSpec) -> dict[str, bytes]:
        """Ask the sidecar to render thumbnails for ``spec``.

        Serializes the spec back to YAML (via ``UnresolvedArenaEnvGraphSpec.to_dict``
        which already unwraps Enums) before shipping it — the sidecar
        re-parses on its end. We round-trip through YAML rather than JSON of
        the dict because the sidecar already imports yaml and we already
        trust ``UnresolvedArenaEnvGraphSpec.from_dict`` to be the canonical parser.

        Returns ``{node_id: png_bytes}`` ready to splice into the HTML.
        Cache-hit nodes read from disk on the parent side (cheap mmap-style
        ``read_bytes``); cache-miss nodes triggered a render in the sidecar
        and we read the freshly-written file by the same code path.
        """
        if not self.is_alive():
            raise SimAppSidecarError("SimApp sidecar is not running — start it first")

        yaml_text = yaml.safe_dump(spec.to_dict(), sort_keys=False)

        with self._lock:
            response = self._request({"cmd": "render_spec", "yaml_text": yaml_text})

        if not response.get("ok"):
            raise SimAppSidecarError(
                f"sidecar render failed: {response.get('error', 'unknown')}\n{response.get('traceback', '')}"
            )

        paths: dict[str, str] = response.get("paths", {}) or {}
        results: dict[str, bytes] = {}
        for node_id, path_str in paths.items():
            path = Path(path_str)
            if path.exists() and path.stat().st_size > 0:
                results[node_id] = path.read_bytes()
            else:
                # Path missing despite a successful response — surface it on
                # stderr but don't bail, the placeholder thumbnail will show.
                print(
                    f"[review_graph]   sidecar reported {node_id} -> {path_str} but file is missing.",
                    file=sys.stderr,
                )
        return results

    def ping(self) -> bool:
        """Cheap liveness check round-trip — returns True on a healthy reply."""
        if not self.is_alive():
            return False
        with self._lock:
            try:
                response = self._request({"cmd": "ping"})
            except SimAppSidecarError:
                return False
        return bool(response.get("ok"))

    # -- internals --

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Single request/response round-trip. Caller owns the lock."""
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        line = json.dumps(payload) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise SimAppSidecarError("sidecar pipe closed unexpectedly") from exc

        reply_line = self._readline_or_die()
        try:
            return json.loads(reply_line)
        except json.JSONDecodeError as exc:
            raise SimAppSidecarError(f"sidecar replied with non-JSON: {reply_line!r}") from exc

    def _readline_or_die(self) -> str:
        """Read a line from sidecar stdout; raise if the pipe closes (sidecar died)."""
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if line == "":
            exit_code = self._proc.poll()
            raise SimAppSidecarError(
                f"sidecar exited prematurely (exit code: {exit_code}). "
                "See its stderr output above for the underlying cause."
            )
        return line

    def _terminate(self) -> None:
        """Hard-kill the sidecar — used when boot fails and graceful is moot."""
        if self._proc is None:
            return
        with contextlib.suppress(Exception):
            self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._proc.kill()
        self._proc = None


# ---------------------------------------------------------------------------
# Top-level HTML
# ---------------------------------------------------------------------------


def _render_html(spec: UnresolvedArenaEnvGraphSpec, thumbnails: dict[str, bytes] | None = None) -> str:
    initial_state = spec.initial_state_spec
    thumbnails = thumbnails or {}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(spec.env_name)} — graph review</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{html_lib.escape(spec.env_name)}</h1>
  <p class="sub">{len(spec.nodes)} nodes · {len(spec.tasks)} tasks · initial state: <code>{html_lib.escape(initial_state.id)}</code></p>
</header>
<main>
  <section class="panel graph-panel">
    <h2>Spatial graph <span class="muted">(initial state: <code>{html_lib.escape(initial_state.id)}</code>)</span></h2>
    <pre class="mermaid">{_render_mermaid(spec, initial_state)}</pre>
    {_render_unary_constraints(initial_state)}
  </section>
  <section class="panel tasks-panel">
    <h2>Tasks</h2>
    {_render_tasks_table(spec)}
  </section>
  <section class="panel nodes-panel">
    <h2>Nodes</h2>
    <div class="node-grid">{_render_node_cards(spec, thumbnails)}</div>
  </section>
</main>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ fontFamily: 'ui-monospace, monospace' }} }});</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Mermaid graph rendering
# ---------------------------------------------------------------------------


def _render_mermaid(spec: UnresolvedArenaEnvGraphSpec, state: ArenaEnvGraphStateSpec) -> str:
    """Emit a left-to-right mermaid graph of spatial and task constraints.

    Binary spatial constraints (reference is set) are drawn as solid edges:
        subject -->|kind| reference

    Unary spatial constraints (no reference) are omitted from the graph and
    listed below it by :func:`_render_unary_constraints` so their params are
    visible.

    Task constraints with a child are drawn as dashed edges:
        parent -.->|type| child

    objectReference nodes are drawn with a dotted edge to their parent node:
        ref_node -. ref .-> parent_node
    """
    lines = ["graph LR"]

    anchor_ids: set[str] = set()
    edge_nodes: set[str] = set()

    # --- Spatial constraints (binary only) ---
    for c in state.spatial_constraints:
        kind = c.kind
        if kind == "is_anchor":
            anchor_ids.add(c.subject)
        if c.reference is not None:
            lines.append(
                f"  {_mermaid_id(c.subject)}[{_mermaid_label(c.subject)}]"
                f" -->|{kind}| "
                f"{_mermaid_id(c.reference)}[{_mermaid_label(c.reference)}]"
            )
            edge_nodes.add(c.subject)
            edge_nodes.add(c.reference)

    # --- Task constraints (dashed edges, binary only) ---
    for tc in state.task_constraints:
        if tc.child is not None:
            lines.append(
                f"  {_mermaid_id(tc.parent)}[{_mermaid_label(tc.parent)}]"
                f" -.->|{_mermaid_label(tc.type.value)}| "
                f"{_mermaid_id(tc.child)}[{_mermaid_label(tc.child)}]"
            )
            edge_nodes.add(tc.parent)
            edge_nodes.add(tc.child)

    # Include every node from the spec so disconnected ones still appear.
    for node in spec.nodes:
        if node.id not in edge_nodes:
            lines.append(f"  {_mermaid_id(node.id)}[{_mermaid_label(node.id)}]")

    # --- objectReference → parent edges (dotted, structural) ---
    # Use bare node IDs (no label re-declaration) — all nodes are already
    # declared above either in constraint edges or in the disconnected-node block.
    nodes_by_id = spec.nodes_by_id
    for node in spec.nodes:
        if node.type.value == "objectReference" and node.parent is not None:
            if node.parent in nodes_by_id:
                lines.append(f"  {_mermaid_id(node.id)} -.->|ref| {_mermaid_id(node.parent)}")

    # Anchor highlight.
    for anchor_id in anchor_ids:
        lines.append(f"  style {_mermaid_id(anchor_id)} fill:#3a7d44,color:#fff,stroke:#7fd17f,stroke-width:2px")

    # Color nodes by type for quick visual scanning.
    type_palette = {
        "background": ("#3a4f7a", "#7aa0d8"),
        "embodiment": ("#7a3a3a", "#d87a7a"),
        "object": ("#7a6b3a", "#d8c47a"),
        "objectReference": ("#6b3a7a", "#c47ad8"),
        "lighting": ("#3a7a7a", "#7ad8d8"),
    }
    for node in spec.nodes:
        if node.id in anchor_ids:
            continue  # anchor style wins
        fill, stroke = type_palette.get(node.type.value, ("#3a3d44", "#888"))
        lines.append(f"  style {_mermaid_id(node.id)} fill:{fill},color:#fff,stroke:{stroke}")

    return "\n".join(lines)


_MERMAID_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _mermaid_id(s: str) -> str:
    """Mermaid node identifiers must be alphanumeric / underscore."""
    return _MERMAID_ID_SAFE.sub("_", s)


def _mermaid_label(s: str) -> str:
    """Escape mermaid-significant characters inside node labels."""
    return s.replace('"', "&quot;").replace("|", "&#124;")


def _render_unary_constraints(state: ArenaEnvGraphStateSpec) -> str:
    """List constraints without a reference below the graph (anchors, position_limits, ...)."""
    rows = []
    for c in state.spatial_constraints:
        if c.reference is not None:
            continue
        params = (
            f' <code class="muted">{html_lib.escape(yaml.safe_dump(c.params, default_flow_style=True).rstrip())}</code>'
            if c.params
            else ""
        )
        rows.append(
            f'<li><span class="badge type-{html_lib.escape(c.kind)}">{html_lib.escape(c.kind)}</span>'
            f" on <code>{html_lib.escape(c.subject)}</code>{params}</li>"
        )
    if not rows:
        return ""
    return (
        f'<details open class="unary"><summary>Unary constraints ({len(rows)})</summary>'
        f'<ul>{"".join(rows)}</ul></details>'
    )


# ---------------------------------------------------------------------------
# Tasks panel
# ---------------------------------------------------------------------------


def _render_tasks_table(spec: UnresolvedArenaEnvGraphSpec) -> str:
    if not spec.tasks:
        return "<p class='muted'><em>No tasks defined.</em></p>"
    rows = []
    for i, t in enumerate(spec.tasks):
        params_str = yaml.safe_dump(t.params, sort_keys=False).rstrip() if t.params else "(empty)"
        desc = html_lib.escape(t.description or "")
        rows.append(
            "<tr>"
            f"<td><code>{i}</code></td>"
            f'<td><span class="badge type-task">{html_lib.escape(t.kind)}</span></td>'
            f"<td>{desc}</td>"
            f"<td><pre>{html_lib.escape(params_str)}</pre></td>"
            "</tr>"
        )
    return (
        "<table class='tasks'>"
        "<thead><tr><th>#</th><th>kind</th><th>description</th><th>params</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Node cards
# ---------------------------------------------------------------------------


def _render_node_cards(spec: UnresolvedArenaEnvGraphSpec, thumbnails: dict[str, bytes]) -> str:
    return "\n".join(_render_one_node_card(node, thumbnails.get(node.id)) for node in spec.nodes)


def _render_one_node_card(node: ArenaEnvGraphNodeSpec, png_bytes: bytes | None) -> str:
    node_dict = node.model_dump(mode="json", exclude_none=True)
    node_yaml = yaml.safe_dump(node_dict, sort_keys=False).rstrip()
    thumb = _render_node_thumbnail(node, png_bytes)
    return f"""<article class="node-card type-{html_lib.escape(node.type.value)}">
  {thumb}
  <div class="node-meta">
    <div class="node-id">{html_lib.escape(node.id)}</div>
    <span class="badge type-{html_lib.escape(node.type.value)}">{html_lib.escape(node.type.value)}</span>
  </div>
  <pre class="node-yaml">{html_lib.escape(node_yaml)}</pre>
</article>"""


def _render_node_thumbnail(node: ArenaEnvGraphNodeSpec, png_bytes: bytes | None = None) -> str:
    """Per-node thumbnail: real USD viewport capture if rendered, else placeholder.

    When ``png_bytes`` is provided (i.e. ``--render-thumbnails`` ran and the
    asset was successfully captured by :func:`_render_thumbnails_for_spec`),
    inline the PNG as a ``data:image/png;base64,...`` URI so the resulting
    HTML is fully self-contained — no sidecar files to keep next to the page.

    Otherwise fall back to the lightweight two-letter placeholder card, so
    a default ``python -m ... review_graph --yaml ...`` invocation still
    produces a useful page without booting Isaac Sim.
    """
    if png_bytes:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return (
            '<div class="thumb thumb-rendered">'
            f'<img src="data:image/png;base64,{b64}" alt="{html_lib.escape(node.name)} thumbnail">'
            f'<span class="thumb-name">{html_lib.escape(node.name)}</span>'
            "</div>"
        )
    initial = (node.name[:2] if node.name else "?").upper()
    return f"""<div class="thumb">
    <span class="thumb-initial">{html_lib.escape(initial)}</span>
    <span class="thumb-name">{html_lib.escape(node.name)}</span>
  </div>"""


# ---------------------------------------------------------------------------
# USD viewport capture (opt-in via --render-thumbnails)
# ---------------------------------------------------------------------------


def _render_thumbnails_with_app(app, spec: UnresolvedArenaEnvGraphSpec) -> dict[str, Path]:
    """Resolve each node's USD via ``AssetRegistry``, render cache-misses, return PNG paths.

    ``app`` must already be a booted ``SimulationApp``. The caller owns the
    lifecycle (Kit may turn ``app.close()`` into ``os._exit(0)`` — that's why
    the sidecar holds the only reference and closes it inside its ``finally``).

    Returns ``{node.id: png_path}`` for nodes whose asset USD could be located
    *and* whose PNG exists on disk (either from the persistent cache under
    ``_THUMBNAIL_CACHE_DIR`` or freshly rendered into the cache by
    :func:`_capture_usd_thumbnails`). Missing entries fall through to the
    placeholder in :func:`_render_node_thumbnail`, so a partial failure (one
    bad asset) never breaks the rest of the page.

    We return ``Path`` rather than ``bytes`` so the sidecar protocol can ship
    just the filenames over its stdin/stdout pipe (a few hundred bytes of JSON
    instead of multiple MB of base64 PNG data). The parent reads the bytes
    itself off the shared filesystem cache.

    Ordering matters: ``SimulationApp`` MUST be launched before any
    ``AssetRegistry`` access, because ``ensure_assets_registered()`` imports
    isaaclab asset modules which transitively load ``pxr``. ``pxr`` loaded
    before ``AppLauncher`` puts Kit's extension manager into an unrecoverable
    state ("extension class wrapper for base class ... has not been created
    yet"). This is the same root cause we fixed for the pytest suite.
    """
    asset_paths = _resolve_node_usd_paths(spec)
    if not asset_paths:
        print("[review_graph] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}

    _THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Split into cache-hits vs to-render. Cache key is sha1(usd_path) so
    # the same USD across multiple envs / nodes hits the same PNG.
    resolved: dict[str, Path] = {}
    to_render: dict[str, tuple[str, Path]] = {}
    for node_id, usd_path in asset_paths.items():
        cache_path = _THUMBNAIL_CACHE_DIR / f"{_usd_cache_key(usd_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[node_id] = cache_path
        else:
            to_render[node_id] = (usd_path, cache_path)

    if to_render:
        print(
            f"[review_graph] rendering {len(to_render)} new thumbnail(s) "
            f"(reusing {len(resolved)} from cache at {_THUMBNAIL_CACHE_DIR})...",
            file=sys.stderr,
        )
        # ``_capture_usd_thumbnails`` still returns ``{node_id: bytes}``, but
        # we only use it as a presence signal here — the same call also wrote
        # the PNG to ``cache_path`` as a side effect, which is what we return.
        captured = _capture_usd_thumbnails(app, to_render)
        for node_id, (_usd_path, cache_path) in to_render.items():
            if node_id in captured and cache_path.exists() and cache_path.stat().st_size > 0:
                resolved[node_id] = cache_path
    else:
        print(f"[review_graph] all {len(resolved)} thumbnail(s) served from cache.", file=sys.stderr)

    return resolved


def _launch_simulation_app():
    """Boot Isaac Sim's ``SimulationApp`` for headless viewport capture, or ``None`` on failure.

    Kept as a tiny helper so the call site can lazy-import inside this
    function — module-level import of ``simulation_app`` would drag Kit
    into every invocation, including ``--help``.
    """
    try:
        # Lazy-import: keeps the default ``review_graph`` invocation Kit-free.
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher  # noqa: PLC0415

        sim_args = argparse.Namespace(headless=True, enable_cameras=True, hide_ui=True, livestream=-1)
        return get_app_launcher(sim_args).app
    except Exception as exc:
        print(f"[review_graph] SimulationApp launch failed: {exc}", file=sys.stderr)
        return None


def _resolve_node_usd_paths(spec: UnresolvedArenaEnvGraphSpec) -> dict[str, str]:
    """Map ``node.id → usd_path`` via :class:`AssetRegistry`, skipping unresolvable nodes.

    Tries two lookup strategies in order:

    1. Class-attribute ``cls.usd_path`` — the convention every ``LibraryObject``
       subclass in ``object_library.py`` follows. No instantiation, cheap.

    2. ``cls().scene_config.robot.spawn.usd_path`` — the convention every
       :class:`EmbodimentBase` subclass uses. Requires instantiating the
       embodiment because the Franka embodiments populate ``scene_config.robot``
       inside ``__init__`` rather than as a class default. Embodiment
       ``__init__`` is light (no Kit / sim required) — it only constructs
       configclass objects.

    This function MUST be called only after ``SimulationApp`` has booted — see
    the docstring of :func:`_render_thumbnails_with_app` for why.
    """
    try:
        from isaaclab_arena.assets.registries import AssetRegistry  # noqa: PLC0415
    except Exception as exc:
        print(f"[review_graph] AssetRegistry import failed: {exc}", file=sys.stderr)
        return {}

    registry = AssetRegistry()
    paths: dict[str, str] = {}
    for node in spec.nodes:
        try:
            if not registry.is_registered(node.name):
                print(f"[review_graph]   {node.id}: asset '{node.name}' not registered, skipping.", file=sys.stderr)
                continue
            cls = registry.get_asset_by_name(node.name)
            usd_path = _extract_usd_path(cls)
            if not usd_path:
                print(f"[review_graph]   {node.id}: '{node.name}' has no usd_path, skipping.", file=sys.stderr)
                continue
            paths[node.id] = usd_path
        except Exception as exc:
            print(f"[review_graph]   {node.id}: lookup failed for '{node.name}': {exc}", file=sys.stderr)
    return paths


def _extract_usd_path(cls) -> str | None:
    """Return the asset's root USD path, or ``None`` if not extractable.

    See :func:`_resolve_node_usd_paths` for the two strategies tried in order.
    """
    # Strategy 1: ``LibraryObject`` convention.
    usd_path = getattr(cls, "usd_path", None)
    if usd_path:
        return usd_path

    # Strategy 2: ``EmbodimentBase`` convention. Walk
    # ``instance.scene_config.robot.spawn.usd_path``. We instantiate with no
    # args; every embodiment ``__init__`` defaults all parameters.
    # NoEmbodiment legitimately has no robot — its instance.scene_config
    # exists but ``.robot`` is absent / None, so the getattr chain returns
    # None and we silently fall through.
    try:
        instance = cls()
    except Exception:
        return None
    scene_config = getattr(instance, "scene_config", None)
    robot = getattr(scene_config, "robot", None) if scene_config is not None else None
    spawn = getattr(robot, "spawn", None) if robot is not None else None
    return getattr(spawn, "usd_path", None) if spawn is not None else None


def _usd_cache_key(usd_path: str) -> str:
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def _capture_usd_thumbnails(app, to_render: dict[str, tuple[str, Path]]) -> dict[str, bytes]:
    """Capture all queued USDs under one already-booted ``SimulationApp``.

    Deduplicates by ``usd_path`` so the same USD shared by multiple nodes is
    only rendered once and the bytes are fanned back out.
    """
    out: dict[str, bytes] = {}

    path_to_node_ids: dict[str, list[str]] = {}
    path_to_cache: dict[str, Path] = {}
    for node_id, (usd_path, cache_path) in to_render.items():
        path_to_node_ids.setdefault(usd_path, []).append(node_id)
        path_to_cache[usd_path] = cache_path

    for usd_path, node_ids in path_to_node_ids.items():
        cache_path = path_to_cache[usd_path]
        try:
            png_bytes = _render_one_usd(app, usd_path, cache_path)
        except Exception as exc:
            print(f"[review_graph]   render failed for {usd_path}: {exc}", file=sys.stderr)
            continue
        if png_bytes:
            for node_id in node_ids:
                out[node_id] = png_bytes

    return out


def _render_one_usd(app, usd_path: str, cache_path: Path) -> bytes | None:
    """Open ``usd_path`` directly as the stage, frame the camera, capture PNG.

    Opening the USD as the stage root (rather than ``new_stage`` + reference
    wrapper) is what makes viewport capture actually produce a file in
    headless mode — Kit's viewport machinery binds to the just-opened stage
    cleanly, whereas a referenced sub-stage left the render product empty in
    every test we tried. The trade-off is that we lose isolation between
    captures (each call replaces the stage), but Kit handles that fine
    because we call ``open_stage`` again on the next asset.
    """
    import omni.usd  # noqa: PLC0415
    from omni.kit.viewport.utility import (  # noqa: PLC0415
        capture_viewport_to_file,
        frame_viewport_prims,
        get_active_viewport,
    )
    from pxr import Sdf  # noqa: PLC0415

    ctx = omni.usd.get_context()
    if not ctx.open_stage(usd_path):
        print(f"[review_graph]   open_stage failed: {usd_path}", file=sys.stderr)
        return None
    stage = ctx.get_stage()

    # Wait for textures / payloads / Nucleus fetches to settle before framing.
    _wait_for_stage_load(app, ctx)

    # Standalone object USDs (avocado, bowl, ...) ship no lights, so a viewport
    # capture renders them as a near-black silhouette against the dark skybox
    # — that's the "blank thumbnail" symptom. Complete scene USDs (maple table)
    # already include their own lighting, so this is a no-op for them.
    _ensure_default_lighting(stage)

    # Use the default prim if present, otherwise the pseudo-root, for framing.
    target_prim = stage.GetDefaultPrim()
    if not target_prim or not target_prim.IsValid():
        target_prim = stage.GetPrimAtPath(Sdf.Path("/"))

    viewport = get_active_viewport()

    # Use Kit's own ``frame_viewport_prims`` (the "F"-key equivalent / ``FramePrimsCommand``)
    # so we go through the viewport camera controller. Manually editing the
    # ``/OmniverseKit_Persp`` xform op directly worked sometimes but Kit's
    # camera controller treats /OmniverseKit_Persp as an internal state and
    # silently overrode our edits for small assets — that's why avocado / bowl
    # captured as tiny specks even with the right math. Letting Kit do the
    # framing is both correct and avoids us re-implementing the math.
    framed = frame_viewport_prims(viewport, prims=[str(target_prim.GetPath())])
    if not framed:
        print(f"[review_graph]   warning: frame_viewport_prims failed for {usd_path}", file=sys.stderr)

    # Settle Hydra after camera change so the captured frame matches the new pose.
    for _ in range(30):
        app.update()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    capture_obj = capture_viewport_to_file(viewport, str(cache_path))

    _wait_for_capture(app, capture_obj, cache_path, max_updates=600)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    print(f"[review_graph]   capture produced no file: {cache_path}", file=sys.stderr)
    return None


def _wait_for_stage_load(app, usd_context, max_updates: int = 600) -> None:
    """Pump frames until ``usd_context.get_stage_loading_status()`` reports nothing pending.

    Returns after stage load completes or after the budget is exhausted. We
    also need a few extra frames after the count goes to zero so material
    binding / texture upload finishes — they don't show up in the load count.
    """
    settled = 0
    for _ in range(max_updates):
        app.update()
        try:
            _msg, loading_count, loaded_count = usd_context.get_stage_loading_status()
        except Exception:
            return
        if loading_count == 0 and loaded_count == 0:
            settled += 1
            if settled > 15:
                return
        else:
            settled = 0


def _wait_for_capture(app, capture_obj, cache_path: Path, max_updates: int = 600) -> None:
    """Pump ``app.update()`` until the capture PNG lands on disk (or we time out).

    Kit's capture future is fulfilled inside its async loop during
    ``app.update()``, but future completion doesn't always coincide with the
    file being flushed — checking the file directly is the most reliable
    completion signal. We also keep the future-based fast path so a
    successful capture doesn't have to wait for the file system to settle.
    """
    if capture_obj is None:
        for _ in range(max_updates):
            app.update()
        return

    future = (
        getattr(capture_obj, "_Capture__future", None)
        or getattr(capture_obj, "_RenderCapture__future", None)
        or getattr(capture_obj, "future", None)
    )

    for _ in range(max_updates):
        app.update()
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return
        if future is not None and future.done():
            # Future is done but file might still be flushing — give it a few frames.
            for _ in range(15):
                app.update()
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    return
            return


def _ensure_default_lighting(stage) -> None:
    """Add a dome + key distant light if the stage has none.

    Without this, standalone object USDs (which don't ship their own lights)
    render as a near-black silhouette. We skip the addition if any
    ``UsdLuxLight``-derived prim already exists on the stage to avoid
    double-lighting scenes like the maple table that bake in their own rig.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: PLC0415

    for prim in stage.Traverse():
        if (
            prim.HasAPI(UsdLux.LightAPI)
            or prim.IsA(UsdLux.BoundableLightBase)
            or prim.IsA(UsdLux.NonboundableLightBase)
        ):
            return

    # Soft hemispherical fill so the asset is visible from any angle, plus a
    # weak directional key for shape definition. Intensities are tuned for
    # OmniPBR / RTX defaults; tweak if asset libraries adopt darker materials.
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/_ReviewDomeLight"))
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/_ReviewKeyLight"))
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(2.0)
    # Aim the key roughly from the camera's 3/4 angle so the lit side faces
    # the viewport.
    key_xformable = UsdGeom.Xformable(key.GetPrim())
    key_xformable.ClearXformOpOrder()
    rot = key_xformable.AddRotateXYZOp()
    rot.Set(Gf.Vec3f(-45.0, 30.0, 0.0))


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #15181d;
  --bg-elev: #1d2128;
  --bg-elev2: #262b34;
  --border: #2f343d;
  --fg: #e4e6eb;
  --fg-muted: #8a9099;
  --accent: #7fd17f;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; font: 14px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--fg); }
header { margin-bottom: 16px; }
header h1 { margin: 0; font-size: 28px; font-weight: 700; }
header .sub { margin: 4px 0 0; color: var(--fg-muted); font-size: 13px; }
main { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto;
       grid-template-areas: "graph nodes" "tasks nodes"; gap: 16px; }
.graph-panel { grid-area: graph; }
.tasks-panel { grid-area: tasks; }
.nodes-panel { grid-area: nodes; }
.panel { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.panel h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
.panel h2 .muted { color: var(--fg-muted); font-weight: 400; font-size: 13px; }
code { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
       background: var(--bg-elev2); padding: 1px 6px; border-radius: 4px; }
pre { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
      background: var(--bg-elev2); padding: 10px 12px; border-radius: 6px; margin: 0;
      white-space: pre-wrap; word-break: break-word; }
.muted { color: var(--fg-muted); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
         font-weight: 600; letter-spacing: 0.03em; background: var(--bg-elev2); color: var(--fg); }
.badge.type-background { background: #3a4f7a; }
.badge.type-embodiment { background: #7a3a3a; }
.badge.type-object { background: #7a6b3a; }
.badge.type-object_reference { background: #6b3a7a; }
.badge.type-lighting { background: #3a7a7a; }
.badge.type-is_anchor { background: #3a7d44; }
.badge.type-position_limits, .badge.type-at_pose, .badge.type-at_position { background: #6b3a7a; }
.badge.type-task { background: #2f343d; border: 1px solid #4a5; color: var(--accent); }
.mermaid { background: var(--bg-elev2); padding: 8px; border-radius: 6px; min-height: 220px;
           display: flex; align-items: center; justify-content: center; }
.unary { margin-top: 12px; }
.unary summary { cursor: pointer; color: var(--fg-muted); font-size: 13px; padding: 4px 0; }
.unary ul { margin: 8px 0 0; padding-left: 20px; list-style: disc; color: var(--fg); }
.unary li { padding: 3px 0; }
table.tasks { width: 100%; border-collapse: collapse; }
table.tasks th, table.tasks td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
                                  vertical-align: top; font-size: 12px; }
table.tasks th { color: var(--fg-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
table.tasks pre { padding: 6px 8px; font-size: 11px; }
.node-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.node-card { background: var(--bg-elev2); border: 1px solid var(--border); border-radius: 8px;
             padding: 12px; display: flex; flex-direction: column; gap: 10px; }
.node-card .thumb { aspect-ratio: 1 / 1; background: linear-gradient(135deg, #2a2f37, #1c2026);
                    border-radius: 6px; display: flex; flex-direction: column;
                    align-items: center; justify-content: center; color: var(--fg-muted);
                    position: relative; overflow: hidden; }
.node-card .thumb-rendered { background: #0e1115; }
.node-card .thumb-rendered img { width: 100%; height: 100%; object-fit: contain; display: block; }
.node-card .thumb-rendered .thumb-name { position: absolute; bottom: 0; left: 0; right: 0;
                                         padding: 4px 6px; background: rgba(15, 17, 21, 0.78);
                                         color: var(--fg); margin: 0; }
.thumb-initial { font-size: 36px; font-weight: 700; color: var(--fg); opacity: 0.6;
                 font-family: ui-monospace, monospace; }
.thumb-name { font-size: 10px; margin-top: 6px; padding: 0 8px; text-align: center; word-break: break-word; }
.node-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.node-id { font-family: ui-monospace, monospace; font-size: 13px; font-weight: 600; word-break: break-all; }
.node-yaml { font-size: 11px; }
"""


if __name__ == "__main__":
    main()
