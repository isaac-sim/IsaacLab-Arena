# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

_MERMAID_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _mermaid_nodes(spec: ArenaEnvGraphSpec) -> list[tuple[str, str]]:
    """Return ``(node_id, role)`` pairs for every asset entry in the spec."""
    nodes = [
        (spec.embodiment.id, "embodiment"),
        (spec.background.id, "background"),
    ]
    nodes.extend((obj.id, "object") for obj in spec.objects)
    nodes.extend((ref.id, "object_reference") for ref in (spec.object_references or []))
    return nodes


def estimate_mermaid_height_px(spec: ArenaEnvGraphSpec) -> int:
    """Heuristic iframe height from node count so typical graphs fit without excessive scrolling."""
    num_nodes = len(_mermaid_nodes(spec))
    return max(260, min(900, 70 * num_nodes))


def render_mermaid_graph(spec: ArenaEnvGraphSpec) -> str:
    """Build left-to-right Mermaid flowchart syntax for spatial and reference edges."""
    lines = ["graph LR"]

    anchor_ids: set[str] = set()
    edge_nodes: set[str] = set()

    for relation in spec.relations:
        kind = relation.kind
        if kind == "is_anchor":
            anchor_ids.add(relation.subject)
        if relation.reference is not None:
            lines.append(
                f"  {_mermaid_id(relation.subject)}[{_mermaid_label(relation.subject)}]"
                f" -->|{_mermaid_label(kind)}| "
                f"{_mermaid_id(relation.reference)}[{_mermaid_label(relation.reference)}]"
            )
            edge_nodes.add(relation.subject)
            edge_nodes.add(relation.reference)

    nodes = _mermaid_nodes(spec)
    spec_node_ids = {node_id for node_id, _role in nodes}
    for node_id, _role in nodes:
        if node_id not in edge_nodes:
            lines.append(f"  {_mermaid_id(node_id)}[{_mermaid_label(node_id)}]")

    for ref in spec.object_references or []:
        if ref.parent_id in spec_node_ids:
            lines.append(f"  {_mermaid_id(ref.id)} -.->|ref| {_mermaid_id(ref.parent_id)}")

    for anchor_id in anchor_ids:
        lines.append(f"  style {_mermaid_id(anchor_id)} fill:#3a7d44,color:#fff,stroke:#7fd17f,stroke-width:2px")

    type_palette = {
        "background": ("#3a4f7a", "#7aa0d8"),
        "embodiment": ("#7a3a3a", "#d87a7a"),
        "object": ("#7a6b3a", "#d8c47a"),
        "object_reference": ("#6b3a7a", "#c47ad8"),
    }
    for node_id, role in nodes:
        if node_id in anchor_ids:
            continue
        fill, stroke = type_palette.get(role, ("#3a3d44", "#888"))
        lines.append(f"  style {_mermaid_id(node_id)} fill:{fill},color:#fff,stroke:{stroke}")

    return "\n".join(lines)


def render_mermaid_html(spec: ArenaEnvGraphSpec) -> str:
    """Render a minimal self-contained HTML document that draws the spatial graph via mermaid.js."""
    syntax = render_mermaid_graph(spec)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body {{ margin: 0; padding: 8px; background: transparent; }}
  .mermaid {{ display: flex; align-items: center; justify-content: center; min-height: 200px; }}
</style>
</head>
<body>
<pre class="mermaid">{syntax}</pre>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ fontFamily: 'ui-monospace, monospace' }} }});</script>
</body>
</html>
"""


def _mermaid_id(value: str) -> str:
    return _MERMAID_ID_SAFE.sub("_", value)


def _mermaid_label(value: str) -> str:
    assert "\n" not in value and "]" not in value, f"Mermaid label contains unexpected characters: {value!r}"
    return value.replace('"', "&quot;").replace("|", "&#124;")
