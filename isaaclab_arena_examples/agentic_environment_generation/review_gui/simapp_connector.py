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

_SIMAPP_CLIENT_SESSION_KEY = "_simapp_client"


def clear_simapp_client() -> None:
    """Disconnect and drop this Streamlit session's SimApp client."""
    client = st.session_state.pop(_SIMAPP_CLIENT_SESSION_KEY, None)
    if client is not None:
        client.disconnect()


def get_simapp_client() -> SimAppClient | None:
    """Return this browser session's client for the SimApp socket from gui_runner."""
    client = st.session_state.get(_SIMAPP_CLIENT_SESSION_KEY)
    if client is not None:
        return client

    socket_path = simapp_socket_from_env()
    if socket_path is None:
        return None
    try:
        client = SimAppClient.connect(socket_path)
    except OSError as exc:
        print(f"[review_gui] SimApp connect failed: {exc}", flush=True)
        return None

    st.session_state[_SIMAPP_CLIENT_SESSION_KEY] = client
    return client


def ensure_simapp() -> SimAppClient | None:
    """Return a healthy SimApp client, reconnecting if this session's client died."""
    client = get_simapp_client()
    if client is not None and client.ping():
        return client

    clear_simapp_client()
    client = get_simapp_client()
    if client is not None and client.ping():
        return client

    clear_simapp_client()
    return None
