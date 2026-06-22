# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit helpers for connecting to the review GUI SimApp server."""

from __future__ import annotations

import streamlit as st

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppClient,
    SimAppError,
    simapp_socket_from_env,
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


def render_dashboard_with_thumbnails(spec: ArenaEnvInitialGraphSpec) -> str:
    """Render review HTML, asking the SimApp server for live USD thumbnails when available."""
    client = ensure_simapp()
    if client is None:
        st.warning(
            "Isaac Sim is unavailable — showing placeholder thumbnails. "
            "Launch the review GUI via gui_runner and check its terminal for errors.",
            icon="⚠️",
        )
        return render_dashboard_html(spec)

    try:
        thumbnails = client.render_spec(spec)
    except SimAppError as exc:
        st.error(
            f"SimApp render failed; showing placeholder thumbnails.\n\n```\n{exc}\n```",
            icon="🛑",
        )
        get_simapp_client.clear()
        return render_dashboard_html(spec)

    return render_dashboard_html(spec, thumbnails=thumbnails if thumbnails else None)
