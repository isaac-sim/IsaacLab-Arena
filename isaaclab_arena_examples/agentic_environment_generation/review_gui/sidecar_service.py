# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Attach to the review GUI SimApp sidecar and render USD thumbnails."""

from __future__ import annotations

from typing import Any

import streamlit as st

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar_client import (
    SimAppSidecarClient,
    SimAppSidecarError,
    sidecar_socket_from_env,
)

_SKIP_REGISTRY_CONTEXT: dict[str, Any] = {"skip_registry": True}


@st.cache_resource(show_spinner=False)
def get_simapp_sidecar() -> SimAppSidecarClient | None:
    """Return a cached client for the sidecar socket exported by gui_runner."""
    socket_path = sidecar_socket_from_env()
    if socket_path is None:
        return None
    try:
        return SimAppSidecarClient.connect(socket_path)
    except OSError as exc:
        print(f"[review_gui] SimApp sidecar connect failed: {exc}", flush=True)
        return None


def ensure_sidecar() -> SimAppSidecarClient | None:
    """Return a healthy sidecar client, reconnecting if the cached session died."""
    sidecar = get_simapp_sidecar()
    if sidecar is not None and sidecar.ping():
        return sidecar
    if sidecar is not None:
        sidecar.close()
    get_simapp_sidecar.clear()
    sidecar = get_simapp_sidecar()
    if sidecar is not None and sidecar.ping():
        return sidecar
    if sidecar is not None:
        sidecar.close()
        get_simapp_sidecar.clear()
    return None


def spec_from_sidecar_dict(spec_dict: dict[str, Any]) -> ArenaEnvInitialGraphSpec:
    """Rebuild a validated spec locally without registry imports."""
    return ArenaEnvInitialGraphSpec.model_validate(spec_dict, context=_SKIP_REGISTRY_CONTEXT)


def render_dashboard_with_thumbnails(spec: ArenaEnvInitialGraphSpec) -> str:
    """Render review HTML, asking the sidecar for live USD thumbnails when available."""
    sidecar = ensure_sidecar()
    if sidecar is None:
        st.warning(
            "Isaac Sim sidecar is unavailable — showing placeholder thumbnails. "
            "Launch the review GUI via gui_runner and check its terminal for errors.",
            icon="⚠️",
        )
        return render_dashboard_html(spec)

    try:
        thumbnails = sidecar.render_spec(spec)
    except SimAppSidecarError as exc:
        st.error(
            f"Sidecar render failed; showing placeholder thumbnails.\n\n```\n{exc}\n```",
            icon="🛑",
        )
        get_simapp_sidecar.clear()
        return render_dashboard_html(spec)

    return render_dashboard_html(spec, thumbnails=thumbnails if thumbnails else None)
