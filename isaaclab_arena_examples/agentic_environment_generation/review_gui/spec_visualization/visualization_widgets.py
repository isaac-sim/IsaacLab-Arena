# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml

import streamlit as st

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import ObjectReferenceSpec, SpatialRelationSpec
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.asset_cards import AssetCard
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.mermaid_graph import (
    estimate_mermaid_height_px,
    render_mermaid_html,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.prim_tree_view import (
    estimate_prim_tree_height_px,
    render_prim_tree_html,
)

_ASSET_GRID_COLS = 4


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
    spec = card.spec
    is_reference = isinstance(spec, ObjectReferenceSpec)
    with st.container(border=True):
        if card.thumbnail_bytes is not None:
            st.image(card.thumbnail_bytes, use_container_width=True)
        elif is_reference and spec.prim_path is None:
            st.caption("⛔ Resolve prim_path to enable collision-mesh snapshot")
        else:
            st.caption("No snapshot available")
        if is_reference:
            note = f"parent `{spec.parent_id}` · prim `{spec.prim_path or '—'}` · {spec.object_type.value}"
        else:
            note = spec.registry_name
        if card.aabb_dimensions_m is not None:
            x, y, z = card.aabb_dimensions_m
            note += f" · [{x:.3f}, {y:.3f}, {z:.3f}]"
        st.markdown(f"**[{card.role}] {spec.id}**")
        st.caption(note)


def _render_asset_grid(cards: list[AssetCard]) -> None:
    """Lay out asset cards in a grid."""
    for start in range(0, len(cards), _ASSET_GRID_COLS):
        columns = st.columns(_ASSET_GRID_COLS)
        for column, card in zip(columns, cards[start : start + _ASSET_GRID_COLS]):
            with column:
                _render_asset_card(card)


def _render_prim_tree(prim_tree: list[UsdPrimRecord]) -> None:
    """Render the background prim tree in a collapsed, searchable, view-only box."""
    if not prim_tree:
        return
    with st.expander("Background prim tree", expanded=False):
        st.components.v1.html(
            render_prim_tree_html(prim_tree),
            height=estimate_prim_tree_height_px(prim_tree),
            scrolling=True,
        )


def render_visualization_widgets(
    spec: ArenaEnvGraphSpec,
    asset_cards: list[AssetCard],
    prim_tree: list[UsdPrimRecord] | None = None,
) -> None:
    """Render the spec visualization (assets, spatial graph, tasks) as native Streamlit widgets."""
    st.markdown(f"**{spec.env_name}**")
    summary = spec.summary()
    if summary:
        st.caption(summary)
    _render_prim_tree(prim_tree or [])
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
