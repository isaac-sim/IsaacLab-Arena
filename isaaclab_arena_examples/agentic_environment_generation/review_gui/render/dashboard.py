# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.mermaid_graph import render_mermaid_graph
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.panels import (
    render_node_cards,
    render_tasks_table,
    render_unary_constraints,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.styles import DASHBOARD_CSS

_SNAPSHOT_LIGHTBOX_HTML = """
<div id="snapshot-lightbox" class="snapshot-lightbox" aria-hidden="true">
  <div class="snapshot-lightbox__content" role="dialog" aria-modal="true" aria-label="Snapshot zoom">
    <div class="snapshot-lightbox__chrome">
      <span id="snapshot-lightbox-title" class="snapshot-lightbox__title"></span>
      <button id="snapshot-lightbox-close" class="snapshot-lightbox__close" type="button">Close</button>
    </div>
    <img id="snapshot-lightbox-img" alt="">
  </div>
</div>
"""

_SNAPSHOT_LIGHTBOX_SCRIPT = """
<script>
(() => {
  const lightbox = document.getElementById("snapshot-lightbox");
  const image = document.getElementById("snapshot-lightbox-img");
  const title = document.getElementById("snapshot-lightbox-title");
  const closeButton = document.getElementById("snapshot-lightbox-close");

  function closeLightbox() {
    lightbox.classList.remove("is-open");
    lightbox.setAttribute("aria-hidden", "true");
    image.removeAttribute("src");
    image.removeAttribute("alt");
    title.textContent = "";
  }

  function openLightbox(source, label) {
    image.src = source;
    image.alt = label ? `${label} snapshot` : "Snapshot";
    title.textContent = label || "Snapshot";
    lightbox.classList.add("is-open");
    lightbox.setAttribute("aria-hidden", "false");
    closeButton.focus();
  }

  document.querySelectorAll(".thumb-zoomable").forEach((thumb) => {
    const activate = () => openLightbox(thumb.dataset.zoomSrc, thumb.dataset.zoomTitle);
    thumb.addEventListener("click", activate);
    thumb.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });

  closeButton.addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && lightbox.classList.contains("is-open")) {
      closeLightbox();
    }
  });
})();
</script>
"""


def render_dashboard_html(
    spec: ArenaEnvGraphSpec,
    thumbnails: dict[str, bytes] | None = None,
    aabb_dimensions_m: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    """Render the self-contained review dashboard HTML for ``spec``."""
    thumbnails = thumbnails or {}
    aabb_dimensions_m = aabb_dimensions_m or {}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(spec.env_name)} — graph review</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>{DASHBOARD_CSS}</style>
</head>
<body>
<header>
  <h1>{html_lib.escape(spec.env_name)}</h1>
  <p class="sub">{html_lib.escape(spec.summary())}</p>
</header>
<main>
  <section class="panel nodes-panel">
    <h2>Assets</h2>
    <div class="node-grid">{render_node_cards(spec, thumbnails, aabb_dimensions_m)}</div>
  </section>
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
{_SNAPSHOT_LIGHTBOX_HTML}
<script>mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ fontFamily: 'ui-monospace, monospace' }} }});</script>
{_SNAPSHOT_LIGHTBOX_SCRIPT}
</body>
</html>
"""
