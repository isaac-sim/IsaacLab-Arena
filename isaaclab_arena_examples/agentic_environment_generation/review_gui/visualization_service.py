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
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppError,
    simapp_socket_from_env,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import (
    clear_simapp_client,
    ensure_simapp,
)


def _spec_render_key(spec: ArenaEnvGraphSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_dashboard_html(spec_key: str) -> str | None:
    cache = st.session_state.get("_dashboard_render_cache")
    if isinstance(cache, dict) and cache.get("key") == spec_key:
        html = cache.get("html")
        if isinstance(html, str):
            return html
    return None


def _store_dashboard_html(spec_key: str, html: str) -> None:
    st.session_state["_dashboard_render_cache"] = {"key": spec_key, "html": html}


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


def render_dashboard_with_thumbnails(spec: ArenaEnvGraphSpec) -> str:
    """Render review HTML, asking the SimApp server for live USD thumbnails when available."""
    spec_key = _spec_render_key(spec)
    cached_html = _cached_dashboard_html(spec_key)
    if cached_html is not None:
        return cached_html

    simapp_expected = simapp_socket_from_env() is not None
    client = ensure_simapp() if simapp_expected else None
    if client is None:
        if simapp_expected:
            _warn_simapp_unavailable_once()
        html = render_dashboard_html(spec)
        _store_dashboard_html(spec_key, html)
        return html

    try:
        thumbnails, aabb_dimensions_m = client.render_spec(spec)
    except SimAppError as exc:
        _show_simapp_render_error_once(exc)
        html = render_dashboard_html(spec)
        _store_dashboard_html(spec_key, html)
        return html
    finally:
        # Release the socket so the sequential SimApp server can accept other tabs.
        clear_simapp_client()

    html = render_dashboard_html(
        spec,
        thumbnails=thumbnails if thumbnails else None,
        aabb_dimensions_m=aabb_dimensions_m or None,
    )
    _store_dashboard_html(spec_key, html)
    return html
