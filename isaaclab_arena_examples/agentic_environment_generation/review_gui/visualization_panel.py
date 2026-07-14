# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml

import streamlit as st

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import SpatialRelationSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import SpecParseResult
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.asset_cards import AssetCard
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.mermaid_graph import (
    estimate_mermaid_height_px,
    render_mermaid_html,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.visualization_service import (
    build_asset_cards_with_thumbnails,
    clear_asset_cards_cache,
    clear_snapshot_render_caches,
)

_ASSET_GRID_COLS = 3


def reset_viz_render_state() -> None:
    """Clear deferred-render bookkeeping so a new spec triggers a fresh preview."""
    st.session_state.pop("_defer_viz_render", None)


def _render_unary_constraints(relations: list[SpatialRelationSpec]) -> None:
    """Render unary spatial relations beside the Mermaid graph."""
    unary = [relation for relation in relations if relation.reference is None]
    st.markdown(f"**Unary constraints** ({len(unary)})")
    if not unary:
        st.caption("No unary constraints.")
        return
    for relation in unary:
        params = f" `{yaml.safe_dump(relation.params, default_flow_style=True).rstrip()}`" if relation.params else ""
        st.markdown(f"- `{relation.kind}` on `{relation.subject}`{params}")


def _render_tasks_table(spec: ArenaEnvGraphSpec) -> None:
    """Render task rows as a native Streamlit dataframe."""
    rows: list[dict[str, str]] = [{
        "#": "root",
        "kind": str(spec.task.composition),
        "description": spec.task.description,
        "params": "(composite root)",
    }]
    for index, task in enumerate(spec.task.subtasks):
        params_str = yaml.safe_dump(task.params, sort_keys=False).rstrip() if task.params else "(empty)"
        rows.append({
            "#": str(index),
            "kind": task.kind,
            "description": "—",
            "params": params_str,
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_asset_card(card: AssetCard) -> None:
    """Render one asset snapshot card; ``st.image`` provides the native fullscreen zoom."""
    with st.container(border=True):
        if card.png_bytes is not None:
            st.image(card.png_bytes, use_container_width=True)
            if card.aabb_dimensions_m is not None:
                x, y, z = card.aabb_dimensions_m
                st.caption(f"AABB {x:.3f} × {y:.3f} × {z:.3f} m")
        else:
            st.caption("No snapshot available")
        st.markdown(f"**{card.node_id}**")
        st.caption(card.role if card.label == card.node_id else f"{card.role} · {card.label}")
        with st.expander("spec"):
            st.code(card.yaml_text, language="yaml")


def _render_asset_grid(cards: list[AssetCard]) -> None:
    """Lay out asset cards in a grid."""
    row: list[AssetCard] = []

    def _flush_row() -> None:
        if not row:
            return
        for column, card in zip(st.columns(_ASSET_GRID_COLS), row):
            with column:
                _render_asset_card(card)
        row.clear()

    for card in cards:
        row.append(card)
        if len(row) == _ASSET_GRID_COLS:
            _flush_row()
    _flush_row()


def render_thumbnail(spec: ArenaEnvGraphSpec, asset_cards: list[AssetCard]) -> None:
    """Render the visualization panel as native Streamlit widgets."""
    st.markdown(f"**{spec.env_name}**")
    summary = spec.summary()
    if summary:
        st.caption(summary)
    if asset_cards:
        st.markdown("**Assets**")
        _render_asset_grid(asset_cards)

    st.markdown("**Spatial graph**")
    graph_col, unary_col = st.columns([3, 1])
    with graph_col:
        st.components.v1.html(
            render_mermaid_html(spec),
            height=estimate_mermaid_height_px(spec),
            scrolling=True,
        )
    with unary_col:
        _render_unary_constraints(spec.relations)

    st.markdown("**Tasks**")
    _render_tasks_table(spec)


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
                st.session_state["rendered_visualization"] = build_asset_cards_with_thumbnails(validation.spec)
            st.session_state["last_rendered_text"] = st.session_state["edited_text"]
            st.toast("Visualization updated.", icon="🔄")

    asset_cards = st.session_state.get("rendered_visualization")
    if isinstance(asset_cards, list):
        st.caption("Updates automatically when the YAML is valid.")
        render_thumbnail(validation.spec, asset_cards)
    elif not st.session_state.get("_defer_viz_render"):
        st.caption("Rendering visualization…")
