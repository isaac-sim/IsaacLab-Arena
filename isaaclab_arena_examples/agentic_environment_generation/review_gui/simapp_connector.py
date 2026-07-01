# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit helpers for connecting to the review GUI SimApp server."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (
    SpecParseResult,
    validate_yaml_text,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppClient,
    SimAppError,
    simapp_socket_from_env,
)

NUM_ENVS = 9
ENV_SPACING_M = 1.5
NUM_STEPS = 10

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


def run_sim_preview_pipeline(
    yaml_text: str,
    *,
    validation: SpecParseResult | None = None,
    num_envs: int = NUM_ENVS,
    num_steps: int = NUM_STEPS,
    env_spacing: float = ENV_SPACING_M,
) -> tuple[bool, str]:
    """Run sim preview via SimApp and store frames in session."""
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
        response = client.run_sim_preview(
            yaml_text,
            num_envs=num_envs,
            num_steps=num_steps,
            env_spacing=env_spacing,
        )
    except SimAppError as exc:
        st.session_state.pop("sim_preview_first", None)
        st.session_state.pop("sim_preview_last", None)
        st.session_state.pop("sim_preview_run_params", None)
        return False, str(exc)
    finally:
        clear_simapp_client()

    try:
        first_path = Path(response["first_frame"])
        last_path = Path(response["last_frame"])
        st.session_state["sim_preview_first"] = first_path.read_bytes()
        st.session_state["sim_preview_last"] = last_path.read_bytes()
        st.session_state["sim_preview_run_params"] = {
            "num_envs": response.get("num_envs", num_envs),
            "num_steps": response.get("num_steps", num_steps),
            "env_spacing": response.get("env_spacing", env_spacing),
        }
    except OSError as exc:
        return False, f"Failed to read preview frames: {exc}"

    return (
        True,
        (
            f"Sim preview complete — {response.get('num_envs', num_envs)} envs, "
            f"{response.get('env_spacing', env_spacing)} m spacing, "
            f"{response.get('num_steps', num_steps)} steps."
        ),
    )
