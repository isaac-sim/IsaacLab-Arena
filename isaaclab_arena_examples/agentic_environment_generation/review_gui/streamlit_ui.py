# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit UI for the initial-graph live editor.

Launch via gui_runner (see that module for CLI usage).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from isaaclab_arena.agentic_environment_generation.spec_io import DEFAULT_AGENTIC_OUTPUT_DIR
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import render_editor_panel
from isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel import (
    DEFAULT_GENERATION_PROMPT,
    get_catalogue_bundle,
    render_generation_panel,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.sim_preview_panel import render_sim_preview_panel
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import (
    ENV_SPACING_M,
    NUM_ENVS,
    NUM_STEPS,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.visualization_panel import (
    render_visualization_panel,
)


def parse_args() -> argparse.Namespace:
    """Parse Streamlit CLI args forwarded after ``--`` by gui_runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env_initial_graph_spec",
        type=Path,
        default=None,
        help="Optional path to an ArenaEnvInitialGraphSpec YAML to open in the editor.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_AGENTIC_OUTPUT_DIR,
        help="Directory for generated initial/linked spec YAML files.",
    )
    return parser.parse_args()


def initialize_state(yaml_path: Path | None, out_dir: Path) -> None:
    """Seed ``st.session_state`` from disk exactly once per session."""
    session_key = str(yaml_path.resolve()) if yaml_path is not None else ""
    if st.session_state.get("_yaml_path") == session_key:
        return

    st.session_state["_yaml_path"] = session_key
    st.session_state.setdefault("generation_prompt", DEFAULT_GENERATION_PROMPT)
    st.session_state.setdefault("editor_version", 0)
    st.session_state.setdefault("sim_preview_num_envs", NUM_ENVS)
    st.session_state.setdefault("sim_preview_num_steps", NUM_STEPS)
    st.session_state.setdefault("sim_preview_env_spacing", ENV_SPACING_M)
    st.session_state["out_dir"] = str(out_dir.resolve())
    st.session_state.pop("_validation_text", None)
    st.session_state.pop("_validation_result", None)

    if yaml_path is None:
        st.session_state["original_text"] = ""
        st.session_state["edited_text"] = ""
        st.session_state["last_rendered_text"] = ""
        st.session_state["rendered_html"] = ""
        st.session_state["save_path"] = ""
        return

    original_text = yaml_path.read_text(encoding="utf-8")

    st.session_state["original_text"] = original_text
    st.session_state["edited_text"] = original_text
    st.session_state["last_rendered_text"] = ""
    st.session_state["rendered_html"] = ""
    st.session_state["save_path"] = str(yaml_path)


def main() -> None:
    """Build the two-column Streamlit layout for generation, editing, and preview."""
    st.set_page_config(
        page_title="ArenaEnvInitialGraphSpec live editor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    args = parse_args()
    yaml_path = args.env_initial_graph_spec.resolve() if args.env_initial_graph_spec is not None else None
    if yaml_path is not None and not yaml_path.exists():
        st.error(f"YAML file not found: {yaml_path}", icon="🛑")
        st.stop()

    initialize_state(yaml_path, args.out_dir.resolve())

    # Populate @st.cache_resource catalogues on first page load so the initial
    # Generate click does not stall on ensure_assets_registered (~10s).
    try:
        get_catalogue_bundle()
    except Exception as exc:
        st.warning(
            f"Asset catalogues could not be loaded: {exc}\n\nGeneration will be unavailable.",
            icon="⚠️",
        )

    st.markdown("### ArenaEnvInitialGraphSpec live editor")
    left, right = st.columns([2, 3], gap="large")
    with left:
        render_generation_panel()
        validation = render_editor_panel(yaml_path)
    with right:
        render_visualization_panel(validation)
        st.divider()
        render_sim_preview_panel(validation)

    # After generation, paint YAML first, then rerun to start SimApp snapshot rendering.
    if st.session_state.pop("_defer_viz_render", False):
        st.rerun()


if __name__ == "__main__":
    main()
