# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import streamlit as st

from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import SpecParseResult
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import run_sim_preview_pipeline


def render_sim_preview_panel(validation: SpecParseResult) -> None:
    """Render full-width sim-preview controls and recorded videos."""
    st.subheader("Sim preview")
    st.caption(
        "Runs a zero-action policy with viewport and embodiment-camera recording. "
        "Environment spacing is the background's largest XY dimension plus 0.5 m."
    )

    preview_cols = st.columns(2)
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
            min_value=1,
            max_value=1000,
            step=1,
            key="sim_preview_num_steps",
            help="Number of zero-action policy steps to record.",
        )

    if st.button(
        "Run relation solver preview",
        type="secondary",
        width="stretch",
        disabled=not validation.is_valid,
        help="Requires valid YAML and a healthy SimApp. This may take several minutes.",
    ):
        with st.spinner(f"Building env, solving relations, and recording {num_steps} steps ({num_envs} envs)…"):
            ok, message = run_sim_preview_pipeline(
                st.session_state["edited_text"],
                validation=validation,
                num_envs=int(num_envs),
                num_steps=int(num_steps),
            )
        if ok:
            st.success(message, icon="✅")
            st.rerun()
        else:
            st.error(f"Sim preview failed\n\n```\n{message}\n```", icon="🛑")

    viewport_video = st.session_state.get("sim_preview_viewport_video")
    camera_videos = st.session_state.get("sim_preview_camera_videos") or []
    run_params = st.session_state.get("sim_preview_run_params") or {}
    if viewport_video:
        st.caption(
            f"Viewport — {run_params.get('num_steps', num_steps)} steps, "
            f"{run_params.get('env_spacing', '?')} m auto spacing"
        )
        st.video(viewport_video)

    if camera_videos:
        st.caption("Embodiment cameras (per environment)")
        video_columns = st.columns(2)
        for index, camera_video in enumerate(
            sorted(camera_videos, key=lambda video: (video["env_id"], video["camera_name"]))
        ):
            with video_columns[index % len(video_columns)]:
                st.caption(f"Env {camera_video['env_id']} — {camera_video['camera_name']}")
                st.video(camera_video["video"])
