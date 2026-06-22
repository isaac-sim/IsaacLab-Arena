# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import streamlit as st

_IFRAME_HEIGHT_PX = 1100


def render_visualization_panel() -> None:
    """Embed the rendered dashboard HTML in the right-hand column."""
    st.subheader("Visualization")
    edited_text = st.session_state.get("edited_text", "").strip()
    if not edited_text:
        st.caption("Generate or enter valid YAML to see the visualization.")
        return

    pending = st.session_state.get("_yaml_before_viz_pass") or edited_text != st.session_state.get(
        "last_rendered_text", ""
    )
    if pending or not st.session_state.get("rendered_html"):
        st.caption("Rendering visualization…")
        return

    st.caption("Updates automatically when the YAML is valid.")
    st.components.v1.html(
        st.session_state["rendered_html"],
        height=_IFRAME_HEIGHT_PX,
        scrolling=True,
    )
