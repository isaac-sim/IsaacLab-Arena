# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit helpers for connecting to the review GUI SimApp server."""

from __future__ import annotations

import streamlit as st

from isaaclab_arena.agentic_tools.inspection.artifacts import consume_result, read_artifact
from isaaclab_arena.agentic_tools.inspection.backend_process import select_socket_path, simapp_socket_from_env
from isaaclab_arena.agentic_tools.inspection.client import InspectorClient
from isaaclab_arena.agentic_tools.inspection.protocol import InspectionError
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (
    SpecParseResult,
    validate_yaml_text,
)

_DEFAULT_BUILDER_CFG = ArenaEnvBuilderCfg()
NUM_ENVS = _DEFAULT_BUILDER_CFG.num_envs
ENV_SPACING_M = _DEFAULT_BUILDER_CFG.env_spacing
NUM_STEPS = 10


def get_inspector_client() -> InspectorClient | None:
    """Build a stateless client from the launcher's backend configuration."""
    if simapp_socket_from_env() is None:
        return None
    return InspectorClient(select_socket_path())


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

    client = get_inspector_client()
    if client is None:
        return False, "SimApp is unavailable — check the gui_runner terminal for boot errors."

    try:
        result = client.run_sim_preview(
            validation.spec,
            num_envs=num_envs,
            num_steps=num_steps,
            env_spacing=env_spacing,
        )
        with consume_result(result):
            video = read_artifact(result, "preview")
        response = result["metadata"]
        st.session_state["sim_preview_video"] = video
        st.session_state["sim_preview_run_params"] = {
            "num_envs": response.get("num_envs", num_envs),
            "num_steps": response.get("num_steps", num_steps),
            "env_spacing": response.get("env_spacing", env_spacing),
        }
    except (InspectionError, OSError) as exc:
        st.session_state.pop("sim_preview_video", None)
        st.session_state.pop("sim_preview_run_params", None)
        return False, str(exc)

    return (
        True,
        (
            f"Sim preview complete — {response.get('num_envs', num_envs)} envs, "
            f"{response.get('env_spacing', env_spacing)} m spacing, "
            f"{response.get('num_steps', num_steps)} steps."
        ),
    )
