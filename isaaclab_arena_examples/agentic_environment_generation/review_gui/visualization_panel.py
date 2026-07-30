# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import streamlit as st

from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import SpecParseResult
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.visualization_widgets import (
    render_visualization_widgets,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.visualization_service import (
    build_asset_cards_with_thumbnails,
    clear_asset_cards_cache,
    clear_snapshot_render_caches,
)


def reset_viz_render_state() -> None:
    """Clear deferred-render bookkeeping so a new spec triggers a fresh preview."""
    st.session_state.pop("_defer_viz_render", None)


def render_visualization_panel(validation: SpecParseResult) -> None:
    """Render the visualization panel in the right column as native Streamlit widgets."""
    st.subheader("Visualization")

    if st.button(
        "Clear cache & rerender",
        help="Delete cached snapshot PNGs on disk and render this spec again.",
    ):
        removed = clear_snapshot_render_caches()
        clear_asset_cards_cache()
        st.session_state["last_rendered_text"] = ""
        st.toast(f"Cleared {removed} cached snapshot(s).", icon="🗑️")
        st.rerun()

    edited_text = st.session_state.get("edited_text", "").strip()
    if not edited_text:
        st.caption("Generate or enter valid YAML to see the visualization.")
        return

    if not validation.is_valid:
        pending = st.session_state["edited_text"] != st.session_state.get("last_rendered_text", "")
        if pending:
            st.session_state["rendered_visualization"] = None
            st.session_state["last_rendered_text"] = st.session_state["edited_text"]
        st.caption("Fix YAML errors to see the visualization.")
        return

    pending = st.session_state["edited_text"] != st.session_state.get("last_rendered_text", "")
    if pending:
        if st.session_state.get("_defer_viz_render"):
            st.caption("Rendering visualization…")
        else:
            with st.spinner("Rendering node snapshots…"):
                asset_cards, prim_tree = build_asset_cards_with_thumbnails(validation.spec)
            st.session_state["rendered_visualization"] = asset_cards
            st.session_state["rendered_prim_tree"] = prim_tree
            st.session_state["last_rendered_text"] = st.session_state["edited_text"]
            st.toast("Visualization updated.", icon="🔄")

    asset_cards = st.session_state.get("rendered_visualization")
    if isinstance(asset_cards, list):
        st.caption("Updates automatically when the YAML is valid.")
        render_visualization_widgets(
            validation.spec,
            asset_cards,
            st.session_state.get("rendered_prim_tree", []),
        )
    elif not st.session_state.get("_defer_viz_render"):
        st.caption("Rendering visualization…")
