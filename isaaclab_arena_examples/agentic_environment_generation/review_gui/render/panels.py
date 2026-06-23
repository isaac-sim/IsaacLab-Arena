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
    """List constraints without a reference beside the spatial graph."""
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
        return '<p class="muted unary-empty"><em>No unary constraints.</em></p>'
    return (
        f'<h3 class="unary-heading">Unary constraints <span class="muted">({len(rows)})</span></h3>'
        f'<ul class="unary-list">{"".join(rows)}</ul>'
    )


def render_tasks_table(spec: ArenaEnvInitialGraphSpec) -> str:
    """Render task rows as an HTML table for the dashboard tasks panel."""
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


def render_node_cards(
    spec: ArenaEnvInitialGraphSpec,
    thumbnails: dict[str, bytes] | None = None,
    aabb_dimensions_m: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    """Render one card per graph node for the dashboard nodes panel."""
    thumbnails = thumbnails or {}
    aabb_dimensions_m = aabb_dimensions_m or {}
    return "\n".join(
        render_node_card(node, thumbnails.get(node.id), aabb_dimensions_m.get(node.id)) for node in spec.nodes
    )


def render_node_card(
    node: ArenaEnvGraphNodeSpec,
    png_bytes: bytes | None = None,
    aabb_dimensions_m: tuple[float, float, float] | None = None,
) -> str:
    """Render a single node card with USD snapshot or placeholder thumbnail and YAML dump."""
    node_dict = node.model_dump(mode="json", exclude_none=True)
    node_yaml = yaml.safe_dump(node_dict, sort_keys=False).rstrip()
    thumb = render_node_thumbnail(node, png_bytes, aabb_dimensions_m)
    return f"""<article class="node-card type-{html_lib.escape(node.type.value)}">
  {thumb}
  <div class="node-meta">
    <div class="node-id">{html_lib.escape(node.id)}</div>
    <span class="badge type-{html_lib.escape(node.type.value)}">{html_lib.escape(node.type.value)}</span>
  </div>
  <pre class="node-yaml">{html_lib.escape(node_yaml)}</pre>
</article>"""
