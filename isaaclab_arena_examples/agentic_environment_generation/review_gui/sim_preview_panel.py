# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import streamlit as st

from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import SpecParseResult
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import run_sim_preview_pipeline

_CAMERA_VIDEO_WIDTH_PX = 250


def _group_camera_videos(camera_videos: list[dict]) -> dict[str, dict[int, bytes]]:
    """Group recorded camera videos by camera name and environment."""
    grouped: dict[str, dict[int, bytes]] = {}
    for video in camera_videos:
        grouped.setdefault(str(video["camera_name"]), {})[int(video["env_id"])] = video["video"]
    return grouped


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
        st.html(f"""
            <style>
            div[class*="st-key-sim-preview-camera-row-"] {{
                display: flex;
                flex-wrap: nowrap !important;
                overflow-x: auto;
                align-items: flex-start;
                padding-bottom: 0.5rem;
            }}
            div[class*="st-key-sim-preview-camera-row-"] > div {{
                flex: 0 0 {_CAMERA_VIDEO_WIDTH_PX}px !important;
                min-width: {_CAMERA_VIDEO_WIDTH_PX}px !important;
            }}
            </style>
            """)
        videos_by_camera = _group_camera_videos(camera_videos)
        recorded_num_envs = int(run_params.get("num_envs", 0))
        for row_index, (camera_name, videos_by_env) in enumerate(sorted(videos_by_camera.items())):
            st.markdown(f"**{camera_name}**")
            with st.container(
                key=f"sim-preview-camera-row-{row_index}",
                horizontal=True,
                gap="small",
            ):
                for env_id in range(recorded_num_envs):
                    with st.container(width=_CAMERA_VIDEO_WIDTH_PX, border=True):
                        st.caption(f"Env {env_id}")
                        video = videos_by_env.get(env_id)
                        if video is None:
                            st.caption("No recording")
                        else:
                            st.video(video, width="stretch")
