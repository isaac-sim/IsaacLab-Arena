# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit helpers for connecting to the review GUI SimApp server."""

from __future__ import annotations

import streamlit as st

from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppClient,
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
