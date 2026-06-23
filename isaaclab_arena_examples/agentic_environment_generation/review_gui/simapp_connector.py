# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit helpers for connecting to the review GUI SimApp server."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html
from isaaclab_arena_examples.agentic_environment_generation.review_gui.sim_preview import (
    ENV_SPACING_M,
    NUM_ENVS,
    NUM_STEPS,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppClient,
    SimAppError,
    simapp_socket_from_env,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.thumbnail_render import (
    resolve_node_aabb_dimensions_m,
)


@st.cache_resource(show_spinner=False)
def get_simapp_client() -> SimAppClient | None:
    """Return a cached client for the SimApp socket exported by gui_runner."""
    socket_path = simapp_socket_from_env()
    if socket_path is None:
        return None
    try:
        return SimAppClient.connect(socket_path)
    except OSError as exc:
        print(f"[review_gui] SimApp connect failed: {exc}", flush=True)
        return None


def ensure_simapp() -> SimAppClient | None:
    """Return a healthy SimApp client, reconnecting if the cached session died."""
    client = get_simapp_client()
    if client is not None and client.ping():
        return client
    if client is not None:
        client.disconnect()
    get_simapp_client.clear()
    client = get_simapp_client()
    if client is not None and client.ping():
        return client
    if client is not None:
        client.disconnect()
        get_simapp_client.clear()
    return None


def _spec_render_key(spec: ArenaEnvInitialGraphSpec) -> str:
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


def render_dashboard_with_thumbnails(spec: ArenaEnvInitialGraphSpec) -> str:
    """Render review HTML, asking the SimApp server for live USD thumbnails when available."""
    spec_key = _spec_render_key(spec)
    cached_html = _cached_dashboard_html(spec_key)
    if cached_html is not None:
        return cached_html

    simapp_expected = simapp_socket_from_env() is not None
    client = ensure_simapp() if simapp_expected else None
    aabb_dimensions_m = resolve_node_aabb_dimensions_m(spec)
    if client is None:
        if simapp_expected:
            _warn_simapp_unavailable_once()
        html = render_dashboard_html(spec, aabb_dimensions_m=aabb_dimensions_m or None)
        _store_dashboard_html(spec_key, html)
        return html

    try:
        thumbnails, aabb_dimensions_m = client.render_spec(spec)
    except SimAppError as exc:
        _show_simapp_render_error_once(exc)
        get_simapp_client.clear()
        return render_dashboard_html(spec, aabb_dimensions_m=aabb_dimensions_m or None)

    html = render_dashboard_html(
        spec,
        thumbnails=thumbnails if thumbnails else None,
        aabb_dimensions_m=aabb_dimensions_m or None,
    )
    _store_dashboard_html(spec_key, html)
    return html


def run_sim_preview_pipeline(yaml_text: str, *, validation=None) -> tuple[bool, str]:
    """Link, build, solve relations, and capture overview frames via SimApp."""
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (  # noqa: PLC0415
        validate_yaml_text,
    )

    if validation is None or not validation.is_valid:
        validation = validate_yaml_text(yaml_text)
        if not validation.is_valid:
            return False, validation.error or "YAML must be valid before running sim preview."

    if simapp_socket_from_env() is None:
        return False, "SimApp is unavailable — launch the review GUI via gui_runner."

    client = ensure_simapp()
    if client is None:
        return False, "SimApp is unavailable — check the gui_runner terminal for boot errors."

    try:
        response = client.run_sim_preview(yaml_text)
    except SimAppError as exc:
        get_simapp_client.clear()
        return False, str(exc)

    try:
        first_path = Path(response["first_frame"])
        last_path = Path(response["last_frame"])
        st.session_state["sim_preview_first"] = first_path.read_bytes()
        st.session_state["sim_preview_last"] = last_path.read_bytes()
    except OSError as exc:
        return False, f"Failed to read preview frames: {exc}"

    return (
        True,
        (
            f"Sim preview complete — {response.get('num_envs', NUM_ENVS)} envs, "
            f"{response.get('env_spacing', ENV_SPACING_M)} m spacing, "
            f"{response.get('num_steps', NUM_STEPS)} steps."
        ),
    )
