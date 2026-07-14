# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate review GUI dashboard rendering with optional SimApp thumbnails."""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.thumbnail_render import (
    ThumbnailRender,
    build_asset_cards,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppError,
    simapp_socket_from_env,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import thumbnail_cache_dir
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import (
    clear_simapp_client,
    ensure_simapp,
)


def _spec_render_key(spec: ArenaEnvGraphSpec) -> str:
    payload = json.dumps({"spec": spec.to_dict()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_dashboard_render(spec_key: str) -> ThumbnailRender | None:
    cache = st.session_state.get("_dashboard_render_cache")
    if isinstance(cache, dict) and cache.get("key") == spec_key:
        render = cache.get("render")
        if isinstance(render, ThumbnailRender):
            return render
    return None


def _store_dashboard_render(spec_key: str, render: ThumbnailRender) -> None:
    st.session_state["_dashboard_render_cache"] = {"key": spec_key, "render": render}


def clear_snapshot_render_caches() -> int:
    """Delete cached review GUI snapshot PNGs and return how many files were removed."""
    removed = 0
    for cache_dir in (thumbnail_cache_dir(),):
        for path in cache_dir.glob("*.png"):
            path.unlink()
            removed += 1
    return removed


def clear_dashboard_render_cache() -> None:
    """Drop the in-memory dashboard HTML cache for the current Streamlit session."""
    st.session_state.pop("_dashboard_render_cache", None)


def _warn_simapp_unavailable_once() -> None:
    if st.session_state.get("_simapp_unavailable_warned"):
        return
    st.session_state["_simapp_unavailable_warned"] = True
    st.warning(
        "Isaac Sim is unavailable — showing placeholder thumbnails. "
        "Check the gui_runner terminal for SimApp boot errors.",
        icon="⚠️",
    )


def _show_simapp_render_error_once(exc: SimAppError) -> None:
    if st.session_state.get("_simapp_render_error_shown"):
        return
    st.session_state["_simapp_render_error_shown"] = True
    st.error(
        f"SimApp render failed; showing placeholder thumbnails.\n\n```\n{exc}\n```",
        icon="🛑",
    )


def render_dashboard_with_thumbnails(spec: ArenaEnvGraphSpec) -> ThumbnailRender:
    """Render the review dashboard, asking the SimApp server for live USD thumbnails when available.

    Returns per-node AssetCards for native Streamlit rendering; graph and tasks are derived from
    ``spec`` at display time in the visualization panel.
    """
    spec_key = _spec_render_key(spec)
    cached = _cached_dashboard_render(spec_key)
    if cached is not None:
        return cached

    thumbnails: dict[str, bytes] = {}
    aabb_dimensions_m: dict[str, tuple[float, float, float]] = {}

    simapp_expected = simapp_socket_from_env() is not None
    client = ensure_simapp() if simapp_expected else None
    if client is None:
        if simapp_expected:
            _warn_simapp_unavailable_once()
    else:
        try:
            thumbnails, aabb_dimensions_m = client.render_spec(spec)
        except SimAppError as exc:
            _show_simapp_render_error_once(exc)
            thumbnails, aabb_dimensions_m = {}, {}
        finally:
            # Release the socket so the sequential SimApp server can accept other tabs.
            clear_simapp_client()

    asset_cards = build_asset_cards(
        spec,
        thumbnails or None,
        aabb_dimensions_m or None,
    )
    render = ThumbnailRender(asset_cards=asset_cards)
    _store_dashboard_render(spec_key, render)
    return render
