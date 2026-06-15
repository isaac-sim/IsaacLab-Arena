# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib
import yaml

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec, ArenaEnvGraphStateSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.thumbnails import render_node_thumbnail


def render_unary_constraints(state: ArenaEnvGraphStateSpec) -> str:
    """List constraints without a reference below the graph (anchors, position_limits, ...)."""
    rows = []
    for constraint in state.spatial_constraints:
        if constraint.reference is not None:
            continue
        params = (
            " <code"
            f' class="muted">{html_lib.escape(yaml.safe_dump(constraint.params, default_flow_style=True).rstrip())}</code>'
            if constraint.params
            else ""
        )
        rows.append(
            f'<li><span class="badge type-{html_lib.escape(constraint.kind)}">{html_lib.escape(constraint.kind)}</span>'
            f" on <code>{html_lib.escape(constraint.subject)}</code>{params}</li>"
        )
    if not rows:
        return ""
    return (
        f'<details open class="unary"><summary>Unary constraints ({len(rows)})</summary>'
        f'<ul>{"".join(rows)}</ul></details>'
    )


def render_tasks_table(spec: ArenaEnvInitialGraphSpec) -> str:
    if not spec.tasks:
        return "<p class='muted'><em>No tasks defined.</em></p>"
    rows = []
    for index, task in enumerate(spec.tasks):
        params_str = yaml.safe_dump(task.params, sort_keys=False).rstrip() if task.params else "(empty)"
        description = html_lib.escape(task.description or "")
        rows.append(
            "<tr>"
            f"<td><code>{index}</code></td>"
            f'<td><span class="badge type-task">{html_lib.escape(task.kind)}</span></td>'
            f"<td>{description}</td>"
            f"<td><pre>{html_lib.escape(params_str)}</pre></td>"
            "</tr>"
        )
    return (
        "<table class='tasks'>"
        "<thead><tr><th>#</th><th>kind</th><th>description</th><th>params</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_node_cards(spec: ArenaEnvInitialGraphSpec, thumbnails: dict[str, bytes] | None = None) -> str:
    thumbnails = thumbnails or {}
    return "\n".join(render_node_card(node, thumbnails.get(node.id)) for node in spec.nodes)


def render_node_card(node: ArenaEnvGraphNodeSpec, png_bytes: bytes | None = None) -> str:
    node_dict = node.model_dump(mode="json", exclude_none=True)
    node_yaml = yaml.safe_dump(node_dict, sort_keys=False).rstrip()
    thumb = render_node_thumbnail(node, png_bytes)
    return f"""<article class="node-card type-{html_lib.escape(node.type.value)}">
  {thumb}
  <div class="node-meta">
    <div class="node-id">{html_lib.escape(node.id)}</div>
    <span class="badge type-{html_lib.escape(node.type.value)}">{html_lib.escape(node.type.value)}</span>
  </div>
  <pre class="node-yaml">{html_lib.escape(node_yaml)}</pre>
</article>"""
