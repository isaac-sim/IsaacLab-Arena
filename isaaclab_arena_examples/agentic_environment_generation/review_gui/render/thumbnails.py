# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import html as html_lib

from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec, ArenaEnvGraphNodeType


def format_aabb_dimensions_m(dims: tuple[float, float, float]) -> str:
    """Format axis-aligned bounding box size as ``x × y × z m``."""
    x, y, z = dims
    return f"{x:.3f} × {y:.3f} × {z:.3f} m"


def _render_aabb_dimensions(aabb_dimensions_m: tuple[float, float, float] | None) -> str:
    if aabb_dimensions_m is None:
        return ""
    return f'<span class="thumb-dims">AABB {html_lib.escape(format_aabb_dimensions_m(aabb_dimensions_m))}</span>'


def _render_unsupported_thumbnail(node: ArenaEnvGraphNodeSpec) -> str:
    return f"""<div class="thumb-wrap">
  <div class="thumb thumb-unsupported">
    <span class="thumb-initial">PR</span>
    <span class="thumb-name">{html_lib.escape(node.name)}</span>
    <span class="thumb-note">Prim reference — snapshot not supported</span>
  </div>
</div>"""


def render_node_thumbnail(
    node: ArenaEnvGraphNodeSpec,
    png_bytes: bytes | None = None,
    aabb_dimensions_m: tuple[float, float, float] | None = None,
) -> str:
    """Per-node thumbnail: USD capture if available, else two-letter placeholder."""
    if node.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
        return _render_unsupported_thumbnail(node)

    dims_html = _render_aabb_dimensions(aabb_dimensions_m)
    if png_bytes:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return (
            '<div class="thumb-wrap">'
            '<div class="thumb thumb-rendered">'
            f'<img src="data:image/png;base64,{b64}" alt="{html_lib.escape(node.name)} thumbnail">'
            f'<span class="thumb-name">{html_lib.escape(node.name)}</span>'
            "</div>"
            f"{dims_html}"
            "</div>"
        )
    initial = (node.name[:2] if node.name else "?").upper()
    return f"""<div class="thumb-wrap">
  <div class="thumb">
    <span class="thumb-initial">{html_lib.escape(initial)}</span>
    <span class="thumb-name">{html_lib.escape(node.name)}</span>
  </div>
  {dims_html}
</div>"""
