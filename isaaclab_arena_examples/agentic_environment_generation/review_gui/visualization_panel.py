# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import streamlit as st

from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import SpecParseResult
from isaaclab_arena_examples.agentic_environment_generation.review_gui.visualization_service import (
    render_dashboard_with_thumbnails,
)

_IFRAME_HEIGHT_PX = 1100

_BROKEN_PLACEHOLDER_HTML = """<!DOCTYPE html><html><body style="
    font-family: ui-monospace, monospace;
    background:#15181d; color:#e4e6eb; padding:24px; margin:0;">
<p>No visualization yet — fix the YAML errors to auto-render.</p>
</body></html>"""


def reset_viz_render_state() -> None:
    """Clear deferred-render bookkeeping so a new spec triggers a fresh preview."""
    st.session_state.pop("_defer_viz_render", None)


def render_visualization_panel(validation: SpecParseResult) -> None:
    """Embed the rendered dashboard HTML in the right column."""
    st.subheader("Visualization")

    edited_text = st.session_state.get("edited_text", "").strip()
    if not edited_text:
        st.caption("Generate or enter valid YAML to see the visualization.")
    elif not validation.is_valid:
        pending = st.session_state["edited_text"] != st.session_state.get("last_rendered_text", "")
        if pending:
            st.session_state["rendered_html"] = _BROKEN_PLACEHOLDER_HTML
            st.session_state["last_rendered_text"] = st.session_state["edited_text"]
        st.caption("Fix YAML errors to see the visualization.")
    else:
        pending = st.session_state["edited_text"] != st.session_state.get("last_rendered_text", "")
        if pending:
            if st.session_state.get("_defer_viz_render"):
                st.caption("Rendering visualization…")
            else:
                with st.spinner("Rendering node snapshots…"):
                    st.session_state["rendered_html"] = render_dashboard_with_thumbnails(validation.spec)
                st.session_state["last_rendered_text"] = st.session_state["edited_text"]
                st.toast("Visualization updated.", icon="🔄")

        html = st.session_state.get("rendered_html", "")
        if html:
            st.caption("Updates automatically when the YAML is valid.")
            st.components.v1.html(
                html,
                height=_IFRAME_HEIGHT_PX,
                scrolling=True,
            )
        elif not st.session_state.get("_defer_viz_render"):
            st.caption("Rendering visualization…")
