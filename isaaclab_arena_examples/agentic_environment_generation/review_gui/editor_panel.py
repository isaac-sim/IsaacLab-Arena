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
from streamlit_ace import st_ace

from isaaclab_arena.agentic_environment_generation.spec_io import env_graph_spec_path, write_env_graph_spec
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec


@dataclass
class SpecParseResult:
    """Outcome of parsing and validating YAML text as an environment graph spec."""

    spec: ArenaEnvGraphSpec | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        """True when a spec was parsed and validated; False for empty or invalid input."""
        return self.spec is not None


def validate_yaml_text(text: str) -> SpecParseResult:
    """Parse YAML text and validate it as an ArenaEnvGraphSpec."""
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
                spec = ArenaEnvGraphSpec.from_dict(raw)
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
            f"Valid spec — {spec.env_name} · {spec.summary()}",
            icon="✅",
        )
    else:
        st.error(f"Invalid YAML\n\n```\n{validation.error}\n```", icon="🛑")


def sync_save_path_from_spec(validation: SpecParseResult) -> None:
    """Point ``save_path`` at the graph-spec YAML path implied by the editor's ``env_name``."""
    if not validation.is_valid:
        return
    out_dir = Path(st.session_state["out_dir"])
    st.session_state["save_path"] = str(env_graph_spec_path(validation.spec.env_name, out_dir))


def try_save_env_graph_spec(spec: ArenaEnvGraphSpec, out_dir: Path) -> tuple[Path | None, str | None]:
    """Write ``spec`` under ``out_dir``.

    Returns:
        ``(path, None)`` on success, or ``(None, error_message)`` on failure.
    """
    try:
        return write_env_graph_spec(spec, out_dir), None
    except OSError as exc:
        return None, f"Save failed: {exc}"
    except Exception:
        return None, traceback.format_exc()


def render_save_button(validation: SpecParseResult) -> None:
    """Render save controls and optional output-directory editor."""
    can_save = validation.is_valid
    assert validation.spec is not None or not can_save
    save_path_str = st.session_state.get("save_path", "")
    save_label = f"Save to {Path(save_path_str).name}" if save_path_str else "Save YAML"
    out_dir = Path(st.session_state["out_dir"])

    if st.button(
        save_label,
        disabled=not can_save,
        use_container_width=True,
        help=f"Writes the validated spec to {out_dir}/<env_name>.yaml.",
    ):
        path, error = try_save_env_graph_spec(validation.spec, out_dir)
        if error is not None:
            st.error(f"Save failed\n\n```\n{error}\n```", icon="🛑")
        else:
            st.session_state["save_path"] = str(path)
            st.session_state["original_text"] = st.session_state["edited_text"]
            st.toast(f"Saved → {path.name}", icon="💾")

    with st.expander("Change output directory", expanded=False):
        out_dir_str = st.session_state["out_dir"]
        new_out_dir = st.text_input(
            "Output directory",
            value=out_dir_str,
            key="out_dir_input",
            help="Directory for generated <env_name>.yaml files.",
        )
        if new_out_dir and new_out_dir != out_dir_str:
            st.session_state["out_dir"] = new_out_dir
            sync_save_path_from_spec(validation)


def render_editor_panel(yaml_path: Path | None) -> SpecParseResult:
    """Render the ACE YAML editor; dashboard preview refreshes in the visualization fragment."""
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
        # ACE returns None on first mount; otherwise sync widget text into session.
        st.session_state["edited_text"] = new_text

    validation = validate_yaml_text(st.session_state["edited_text"])
    render_validation_badge(validation)
    sync_save_path_from_spec(validation)

    render_save_button(validation)
    return validation
