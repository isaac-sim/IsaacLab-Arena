# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.mermaid_graph import render_mermaid_graph
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.panels import (
    AssetCard,
    render_tasks_table,
    render_unary_constraints,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.styles import DASHBOARD_CSS


@dataclass(frozen=True)
class DashboardRender:
    """Split dashboard output: asset cards rendered as native Streamlit widgets, the rest as embedded HTML."""

    html: str
    """Self-contained HTML for the spatial graph and tasks panels (embedded via st.components)."""
    asset_cards: list[AssetCard] = field(default_factory=list)
    """Per-node snapshot cards rendered natively so they get Streamlit's fullscreen zoom."""


def render_dashboard_html(spec: ArenaEnvGraphSpec) -> str:
    """Render the graph + tasks portion of the review dashboard as self-contained HTML.

    The Assets snapshots are rendered separately as native Streamlit widgets (see AssetCard) so
    they benefit from Streamlit's built-in fullscreen zoom, matching the sim preview frames.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>graph review</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>{DASHBOARD_CSS}</style>
</head>
<body>
<main>
  <section class="panel graph-panel">
    <h2>Spatial graph</h2>
    <div class="graph-row">
      <div class="graph-mermaid">
        <pre class="mermaid">{render_mermaid_graph(spec)}</pre>
      </div>
      <aside class="graph-unary">
        {render_unary_constraints(spec.relations)}
      </aside>
    </div>
  </section>
  <section class="panel tasks-panel">
    <h2>Tasks</h2>
    {render_tasks_table(spec)}
  </section>
</main>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ fontFamily: 'ui-monospace, monospace' }} }});</script>
</body>
</html>
"""
