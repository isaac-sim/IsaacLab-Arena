# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit UI for the initial-graph live editor.

Launch via :mod:`~isaaclab_arena_examples.agentic_environment_generation.review_gui.server`:

    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.review_gui.server \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml
"""

from __future__ import annotations

import argparse
import traceback
import yaml
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.html_document import render_html_for_spec

# Visualization iframe height. Tuned so the graph + tasks + node grid all
# fit without an outer Streamlit scrollbar swallowing the inner one.
_IFRAME_HEIGHT_PX = 1100

_BROKEN_PLACEHOLDER_HTML = """<!DOCTYPE html><html><body style="
    font-family: ui-monospace, monospace;
    background:#15181d; color:#e4e6eb; padding:24px; margin:0;">
<p>No visualization yet — fix the YAML errors to auto-render.</p>
</body></html>"""


@dataclass
class ValidationResult:
    """Outcome of parsing and validating YAML text as an initial graph spec."""

    spec: ArenaEnvInitialGraphSpec | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.spec is not None


def validate_yaml_text(text: str) -> ValidationResult:
    """Parse ``text`` as YAML and validate it as an :class:`ArenaEnvInitialGraphSpec`."""
    try:
        raw = yaml.safe_load(text)
        spec = ArenaEnvInitialGraphSpec.model_validate(raw)
        return ValidationResult(spec=spec, error=None)
    except Exception:
        return ValidationResult(spec=None, error=traceback.format_exc())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Path to the ArenaEnvInitialGraphSpec YAML to open in the editor.",
    )
    return parser.parse_args()


def initialize_state(yaml_path: Path) -> None:
    """Seed ``st.session_state`` from disk exactly once per session."""
    if st.session_state.get("_yaml_path") == str(yaml_path):
        return

    original_text = yaml_path.read_text(encoding="utf-8")

    st.session_state["_yaml_path"] = str(yaml_path)
    st.session_state["original_text"] = original_text
    st.session_state["edited_text"] = original_text
    st.session_state["last_rendered_text"] = original_text
    st.session_state["save_path"] = str(yaml_path)

    initial = validate_yaml_text(original_text)
    if not initial.is_valid:
        st.session_state["rendered_html"] = _BROKEN_PLACEHOLDER_HTML
    else:
        st.session_state["rendered_html"] = render_html_for_spec(initial.spec)


def render_validation_badge(validation: ValidationResult) -> None:
    if validation.is_valid:
        spec = validation.spec
        st.success(
            f"Valid spec — {spec.env_name} · {len(spec.nodes)} nodes · "
            f"{len(spec.tasks)} tasks · initial state: {spec.initial_state_spec.id}",
            icon="✅",
        )
    else:
        st.error(f"Invalid YAML\n\n```\n{validation.error}\n```", icon="🛑")


def render_save_button(validation: ValidationResult) -> None:
    can_save = validation.is_valid
    save_path_str = st.session_state["save_path"]

    if st.button(
        f"Save to {Path(save_path_str).name}",
        disabled=not can_save,
        use_container_width=True,
        help=f"Writes the editor contents to {save_path_str}. Disabled while YAML is invalid.",
    ):
        try:
            Path(save_path_str).write_text(st.session_state["edited_text"], encoding="utf-8")
            st.session_state["original_text"] = st.session_state["edited_text"]
            st.toast(f"Saved → {save_path_str}", icon="💾")
        except OSError as exc:
            st.error(f"Save failed: {exc}", icon="🛑")

    with st.expander("Change save location", expanded=False):
        new_path = st.text_input(
            "Save path",
            value=save_path_str,
            key="save_path_input",
            help="Defaults to the YAML file passed via --yaml.",
        )
        if new_path and new_path != save_path_str:
            st.session_state["save_path"] = new_path


def render_editor_panel(yaml_path: Path) -> ValidationResult:
    try:
        from streamlit_ace import st_ace  # noqa: PLC0415
    except ImportError as exc:
        st.error(
            "`streamlit-ace` is not installed. Inside the isaaclab_arena container run:\n"
            "`python -m pip install --user --ignore-installed streamlit-ace`\n\n"
            f"Underlying error: {exc}",
            icon="🛑",
        )
        st.stop()

    st.subheader("YAML editor")
    st.caption(f"Source: `{yaml_path}`")

    new_text = st_ace(
        value=st.session_state["edited_text"],
        language="yaml",
        theme="monokai",
        keybinding="vscode",
        font_size=13,
        tab_size=2,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=False,
        min_lines=30,
        key=f"ace_editor::{yaml_path}",
    )
    if new_text is not None:
        st.session_state["edited_text"] = new_text

    validation = validate_yaml_text(st.session_state["edited_text"])
    render_validation_badge(validation)

    edited_since_render = st.session_state["edited_text"] != st.session_state["last_rendered_text"]
    if validation.is_valid and edited_since_render:
        with st.spinner("Rendering visualization…"):
            st.session_state["rendered_html"] = render_html_for_spec(validation.spec)
        st.session_state["last_rendered_text"] = st.session_state["edited_text"]
        st.toast("Visualization updated.", icon="🔄")

    render_save_button(validation)
    return validation


def render_visualization_panel() -> None:
    st.subheader("Visualization")
    st.caption("Updates automatically when the YAML is valid.")

    st.components.v1.html(
        st.session_state["rendered_html"],
        height=_IFRAME_HEIGHT_PX,
        scrolling=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="ArenaEnvInitialGraphSpec live editor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    args = parse_args()
    yaml_path = args.yaml.resolve()
    if not yaml_path.exists():
        st.error(f"YAML file not found: {yaml_path}", icon="🛑")
        st.stop()

    initialize_state(yaml_path)

    st.markdown("### ArenaEnvInitialGraphSpec live editor")
    left, right = st.columns([2, 3], gap="large")
    with left:
        render_editor_panel(yaml_path)
    with right:
        render_visualization_panel()


# Streamlit invokes the script top-level on every rerun.
main()
