# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import html as html_lib

from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec


def render_node_thumbnail(node: ArenaEnvGraphNodeSpec, png_bytes: bytes | None = None) -> str:
    """Per-node thumbnail: USD capture if available, else two-letter placeholder."""
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
