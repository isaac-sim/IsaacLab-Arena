# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimApp sidecar lifecycle and thumbnail rendering for the review GUI."""

from __future__ import annotations

import atexit
from typing import Any

import streamlit as st

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar_client import (
    SimAppSidecar,
    SimAppSidecarError,
)

_SKIP_REGISTRY_CONTEXT: dict[str, Any] = {"skip_registry": True}


@st.cache_resource(show_spinner="Booting Isaac Sim sidecar (≈30s first run, cached afterwards)…")
def get_simapp_sidecar() -> SimAppSidecar | None:
    """Spawn the SimApp sidecar once per Streamlit server process."""
    sidecar = SimAppSidecar()
    try:
        sidecar.start()
    except SimAppSidecarError as exc:
        print(f"[review_gui] SimApp sidecar failed to start: {exc}", flush=True)
        return None

    atexit.register(sidecar.close)
    return sidecar


def ensure_sidecar() -> SimAppSidecar | None:
    """Return a healthy sidecar, re-spawning if the cached one died."""
    sidecar = get_simapp_sidecar()
    if sidecar is not None and sidecar.is_alive():
        return sidecar
    if sidecar is not None:
        sidecar.close()
    get_simapp_sidecar.clear()
    return get_simapp_sidecar()


def spec_from_sidecar_dict(spec_dict: dict[str, Any]) -> ArenaEnvInitialGraphSpec:
    """Rebuild a validated spec locally without registry imports."""
    return ArenaEnvInitialGraphSpec.model_validate(spec_dict, context=_SKIP_REGISTRY_CONTEXT)


def render_dashboard_with_thumbnails(spec: ArenaEnvInitialGraphSpec) -> str:
    """Render review HTML, asking the sidecar for live USD thumbnails when available."""
    sidecar = ensure_sidecar()
    if sidecar is None:
        st.warning(
            "Isaac Sim sidecar is unavailable — showing placeholder thumbnails. "
            "Check the terminal where you launched the server for the underlying error.",
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
        with st.spinner("Resetting the SimApp sidecar…"):
            get_simapp_sidecar.clear()
        return render_dashboard_html(spec)

    return render_dashboard_html(spec, thumbnails=thumbnails if thumbnails else None)
