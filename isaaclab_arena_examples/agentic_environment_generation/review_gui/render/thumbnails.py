# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib

from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec


# TODO(qianl): Replace placeholder thumbnails with sim-rendered snapshots .
def render_placeholder_thumbnail(node: ArenaEnvGraphNodeSpec) -> str:
    """Per-node placeholder thumbnail — two-letter initial."""
    initial = (node.name[:2] if node.name else "?").upper()
    return f"""<div class="thumb">
    <span class="thumb-initial">{html_lib.escape(initial)}</span>
    <span class="thumb-name">{html_lib.escape(node.name)}</span>
  </div>"""
