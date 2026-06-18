# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import traceback
import yaml
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html

DEFAULT_SAVE_PATH = "isaaclab_arena_environments/agent_generated/generated_spec.yaml"

_BROKEN_PLACEHOLDER_HTML = """<!DOCTYPE html><html><body style="
    font-family: ui-monospace, monospace;
    background:#15181d; color:#e4e6eb; padding:24px; margin:0;">
<p>No visualization yet — fix the YAML errors to auto-render.</p>
</body></html>"""


@dataclass
class SpecParseResult:
    """Outcome of parsing and validating YAML text as an initial graph spec."""

    spec: ArenaEnvInitialGraphSpec | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        """True when a spec was parsed and validated; False for empty or invalid input."""
        return self.spec is not None


def validate_yaml_text(text: str) -> SpecParseResult:
    """Parse ``text`` as YAML and validate it as an :class:`ArenaEnvInitialGraphSpec`."""
    cached_text = st.session_state.get("_validation_text")
    cached_result = st.session_state.get("_validation_result")
    if cached_text == text and isinstance(cached_result, SpecParseResult):
        return cached_result

    if not text.strip():
        result = SpecParseResult(spec=None, error=None)
    else:
        try:
            raw = yaml.safe_load(text)
            if raw is None:
                result = SpecParseResult(spec=None, error="YAML is empty")
            elif not isinstance(raw, dict):
                result = SpecParseResult(spec=None, error=f"Expected mapping, got {type(raw).__name__}")
            else:
                spec = ArenaEnvInitialGraphSpec.from_dict(raw)
                result = SpecParseResult(spec=spec, error=None)
        except Exception:
            result = SpecParseResult(spec=None, error=traceback.format_exc())

    st.session_state["_validation_text"] = text
    st.session_state["_validation_result"] = result
    return result


def render_validation_badge(validation: SpecParseResult) -> None:
    """Show a success or error badge for the current editor YAML."""
    if validation.spec is None and validation.error is None:
        return
    if validation.is_valid:
        spec = validation.spec
        st.success(
            f"Valid spec — {spec.env_name} · {len(spec.nodes)} nodes · "
            f"{len(spec.tasks)} tasks · initial state: {spec.initial_state_spec.id}",
            icon="✅",
        )
    else:
        st.error(f"Invalid YAML\n\n```\n{validation.error}\n```", icon="🛑")


def render_save_button(validation: SpecParseResult) -> None:
    """Render save controls and optional save-path editor."""
    can_save = validation.is_valid
    save_path_str = st.session_state["save_path"]
    save_label = f"Save to {Path(save_path_str).name}" if save_path_str else "Save YAML"

    if st.button(
        save_label,
        disabled=not can_save,
        use_container_width=True,
        help=f"Writes the editor contents to {save_path_str}. Disabled while YAML is invalid.",
    ):
        try:
            out_path = Path(save_path_str)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(st.session_state["edited_text"], encoding="utf-8")
            st.session_state["original_text"] = st.session_state["edited_text"]
            st.toast(f"Saved → {save_path_str}", icon="💾")
        except OSError as exc:
            st.error(f"Save failed: {exc}", icon="🛑")

    with st.expander("Change save location", expanded=False):
        new_path = st.text_input(
            "Save path",
            value=save_path_str,
            key="save_path_input",
            help=(
                "Defaults to the file passed via --env_initial_graph_spec, or a generated-spec path when none was"
                " given."
            ),
        )
        if new_path and new_path != save_path_str:
            st.session_state["save_path"] = new_path


def render_editor_panel(yaml_path: Path | None) -> SpecParseResult:
    """Render the ACE YAML editor and refresh the dashboard when text changes."""
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
    if yaml_path is not None:
        st.caption(f"Source: `{yaml_path}`")
    else:
        st.caption("No file loaded — generate a spec or paste YAML.")

    editor_key = str(yaml_path) if yaml_path is not None else "new"
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
        key=f"ace_editor::{editor_key}::{st.session_state.get('editor_version', 0)}",
    )
    if new_text is not None:
        st.session_state["edited_text"] = new_text

    validation = validate_yaml_text(st.session_state["edited_text"])
    render_validation_badge(validation)

    edited_since_render = st.session_state["edited_text"] != st.session_state["last_rendered_text"]
    if edited_since_render:
        if validation.is_valid:
            with st.spinner("Rendering visualization…"):
                st.session_state["rendered_html"] = render_dashboard_html(validation.spec)
        else:
            st.session_state["rendered_html"] = _BROKEN_PLACEHOLDER_HTML
        st.session_state["last_rendered_text"] = st.session_state["edited_text"]
        if validation.is_valid:
            st.toast("Visualization updated.", icon="🔄")

    render_save_button(validation)
    return validation
