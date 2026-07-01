# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import streamlit as st

from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import SpecParseResult
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import run_sim_preview_pipeline


def render_sim_preview_panel(validation: SpecParseResult) -> None:
    """Sim-preview controls and viewport frame display in the right column."""
    st.subheader("Sim preview")
    st.caption(
        "Runs link → to_arena_env → relation solver, then zero-action steps. "
        "Viewport captures use a world-frame overview of the full env grid."
    )

    preview_cols = st.columns(3)
    with preview_cols[0]:
        num_envs = st.number_input(
            "Parallel envs",
            min_value=1,
            max_value=256,
            step=1,
            key="sim_preview_num_envs",
            help="Number of cloned environments in the preview rollout.",
        )
    with preview_cols[1]:
        num_steps = st.number_input(
            "Zero-action steps",
            min_value=0,
            max_value=1000,
            step=1,
            key="sim_preview_num_steps",
            help="Number of zero-action policy steps after reset before the second capture.",
        )
    with preview_cols[2]:
        env_spacing = st.number_input(
            "Env spacing (m)",
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            format="%.1f",
            key="sim_preview_env_spacing",
            help="Spacing between cloned environments in the preview grid.",
        )

    if st.button(
        "Run link + relation solver preview",
        type="secondary",
        use_container_width=True,
        disabled=not validation.is_valid,
        help="Requires valid YAML and a healthy SimApp. This may take several minutes.",
    ):
        with st.spinner(
            f"Building env, solving relations, and rolling out {num_steps} steps ({num_envs} envs @ {env_spacing} m)…"
        ):
            ok, message = run_sim_preview_pipeline(
                st.session_state["edited_text"],
                validation=validation,
                num_envs=int(num_envs),
                num_steps=int(num_steps),
                env_spacing=float(env_spacing),
            )
        if ok:
            st.success(message, icon="✅")
            st.rerun()
        else:
            st.error(f"Sim preview failed\n\n```\n{message}\n```", icon="🛑")

    first_frame = st.session_state.get("sim_preview_first")
    last_frame = st.session_state.get("sim_preview_last")
    run_params = st.session_state.get("sim_preview_run_params") or {}
    displayed_steps = int(run_params.get("num_steps", num_steps))
    if first_frame and last_frame:
        frame_cols = st.columns(2)
        with frame_cols[0]:
            st.caption("Viewport — frame 1 (after reset)")
            st.image(first_frame, use_container_width=True)
        with frame_cols[1]:
            st.caption(f"Viewport — frame 2 (after {displayed_steps} zero-action steps)")
            st.image(last_frame, use_container_width=True)
