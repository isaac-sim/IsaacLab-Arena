# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib
import yaml

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec, SpatialRelationSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.thumbnails import render_asset_thumbnail


def render_unary_constraints(relations: list[SpatialRelationSpec]) -> str:
    """List unary relations beside the spatial graph."""
    rows = []
    for relation in relations:
        if relation.reference is not None:
            continue
        params = (
            " <code"
            f' class="muted">{html_lib.escape(yaml.safe_dump(relation.params, default_flow_style=True).rstrip())}</code>'
            if relation.params
            else ""
        )
        rows.append(
            f'<li><span class="badge type-{html_lib.escape(relation.kind)}">{html_lib.escape(relation.kind)}</span>'
            f" on <code>{html_lib.escape(relation.subject)}</code>{params}</li>"
        )
    if not rows:
        return '<p class="muted unary-empty"><em>No unary constraints.</em></p>'
    return (
        f'<h3 class="unary-heading">Unary constraints <span class="muted">({len(rows)})</span></h3>'
        f'<ul class="unary-list">{"".join(rows)}</ul>'
    )


def render_tasks_table(spec: ArenaEnvGraphSpec) -> str:
    """Render task rows as an HTML table for the dashboard tasks panel."""
    atomic_tasks = spec.task.subtasks
    rows = []
    composition = html_lib.escape(spec.task.composition)
    summary = html_lib.escape(spec.task.description)
    rows.append(
        "<tr>"
        "<td><code>root</code></td>"
        f'<td><span class="badge type-task">{composition}</span></td>'
        f"<td>{summary}</td>"
        "<td><pre>(composite root)</pre></td>"
        "</tr>"
    )
    for index, task in enumerate(atomic_tasks):
        params_str = yaml.safe_dump(task.params, sort_keys=False).rstrip() if task.params else "(empty)"
        rows.append(
            "<tr>"
            f"<td><code>{index}</code></td>"
            f'<td><span class="badge type-task">{html_lib.escape(task.kind)}</span></td>'
            "<td class='muted'><em>—</em></td>"
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
    spec: ArenaEnvGraphSpec,
    thumbnails: dict[str, bytes] | None = None,
    aabb_dimensions_m: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    """Render one card per asset for the dashboard nodes panel."""
    thumbnails = thumbnails or {}
    aabb_dimensions_m = aabb_dimensions_m or {}
    entries = [
        ("embodiment", spec.embodiment),
        ("background", spec.background),
        *(("object", obj) for obj in spec.objects),
    ]
    return "\n".join(
        render_node_card(role, asset, thumbnails.get(asset.id), aabb_dimensions_m.get(asset.id))
        for role, asset in entries
    )


def render_node_card(
    role: str,
    asset: AssetSpec,
    png_bytes: bytes | None = None,
    aabb_dimensions_m: tuple[float, float, float] | None = None,
) -> str:
    """Render a single asset card with USD snapshot or placeholder thumbnail and YAML dump."""
    node_yaml = yaml.safe_dump(asset.model_dump(mode="json", exclude_none=True), sort_keys=False).rstrip()
    thumb = render_asset_thumbnail(asset.registry_name, png_bytes, aabb_dimensions_m)
    return f"""<article class="node-card type-{html_lib.escape(role)}">
  {thumb}
  <div class="node-meta">
    <div class="node-id">{html_lib.escape(asset.id)}</div>
    <span class="badge type-{html_lib.escape(role)}">{html_lib.escape(role)}</span>
  </div>
  <pre class="node-yaml">{html_lib.escape(node_yaml)}</pre>
</article>"""
