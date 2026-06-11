# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ArenaEnvInitialGraphSpec review tool — Streamlit live editor.

The CLI is a thin launcher: it boots the Streamlit app in ``review_app.py``.

The tool accepts an ``ArenaEnvInitialGraphSpec`` YAML as input. The user
edits the spec directly inside the Streamlit editor and the preview updates
in real time.

Three panels (dark dashboard style) inside the embedded view:
  * Top-left — graph diagram (mermaid.js, CDN-loaded) of the initial-state
    spatial constraints. Anchor nodes are highlighted; constraints without
    a reference (is_anchor / position_limits / at_pose / ...) are listed below
    the graph rather than rendered as self-loops.
  * Bottom-left — task table (index, kind, description, params).
  * Right — node card grid: type badge, asset name, and the per-node YAML
    stanza.

Usage:
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.review_graph \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml

    # Custom port:
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.review_graph \\
        --yaml <path> --port 8600

Public API used by ``review_app.py``:
    * :func:`render_html_for_spec` — full HTML payload for a spec.
"""

from __future__ import annotations

import argparse
import html as html_lib
import os
import re
import subprocess
import sys
import yaml
from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec, ArenaEnvGraphStateSpec


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
        help="Path to an ArenaEnvInitialGraphSpec YAML file. The Streamlit app will open it for live editing.",
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

    We resolve ``review_app.py`` next to this file rather than going through
    ``-m`` so Streamlit picks the path up cleanly (``streamlit run`` doesn't
    accept module dotted-paths).
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


def render_html_for_spec(spec: ArenaEnvInitialGraphSpec) -> str:
    """Render the review HTML for ``spec`` with placeholder node thumbnails.

    Thin public alias of :func:`_render_html` so external entry points don't
    have to reach into a private name.
    """
    return _render_html(spec)


# ---------------------------------------------------------------------------
# Top-level HTML
# ---------------------------------------------------------------------------


def _render_html(spec: ArenaEnvInitialGraphSpec) -> str:
    initial_state = spec.initial_state_spec
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
    <div class="node-grid">{_render_node_cards(spec)}</div>
  </section>
</main>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ fontFamily: 'ui-monospace, monospace' }} }});</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Mermaid graph rendering
# ---------------------------------------------------------------------------


def _render_mermaid(spec: ArenaEnvInitialGraphSpec, state: ArenaEnvGraphStateSpec) -> str:
    """Emit a left-to-right mermaid graph of spatial and task constraints.

    Binary spatial constraints (reference is set) are drawn as solid edges:
        subject -->|kind| reference

    Unary spatial constraints (no reference) are omitted from the graph and
    listed below it by :func:`_render_unary_constraints` so their params are
    visible.

    Task constraints with a child are drawn as dashed edges:
        parent -.->|type| child

    object_reference nodes are drawn with a dotted edge to their parent node:
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

    # --- object_reference → parent edges (dotted, structural) ---
    # Use bare node IDs (no label re-declaration) — all nodes are already
    # declared above either in constraint edges or in the disconnected-node block.
    nodes_by_id = spec.nodes_by_id
    for node in spec.nodes:
        if node.type.value == "object_reference" and node.parent is not None:
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
        "object_reference": ("#6b3a7a", "#c47ad8"),
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


def _render_tasks_table(spec: ArenaEnvInitialGraphSpec) -> str:
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


def _render_node_cards(spec: ArenaEnvInitialGraphSpec) -> str:
    return "\n".join(_render_one_node_card(node) for node in spec.nodes)


def _render_one_node_card(node: ArenaEnvGraphNodeSpec) -> str:
    node_dict = node.model_dump(mode="json", exclude_none=True)
    node_yaml = yaml.safe_dump(node_dict, sort_keys=False).rstrip()
    thumb = _render_node_thumbnail(node)
    return f"""<article class="node-card type-{html_lib.escape(node.type.value)}">
  {thumb}
  <div class="node-meta">
    <div class="node-id">{html_lib.escape(node.id)}</div>
    <span class="badge type-{html_lib.escape(node.type.value)}">{html_lib.escape(node.type.value)}</span>
  </div>
  <pre class="node-yaml">{html_lib.escape(node_yaml)}</pre>
</article>"""


def _render_node_thumbnail(node: ArenaEnvGraphNodeSpec) -> str:
    """Per-node placeholder thumbnail — two-letter initial."""
    initial = (node.name[:2] if node.name else "?").upper()
    return f"""<div class="thumb">
    <span class="thumb-initial">{html_lib.escape(initial)}</span>
    <span class="thumb-name">{html_lib.escape(node.name)}</span>
  </div>"""


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
