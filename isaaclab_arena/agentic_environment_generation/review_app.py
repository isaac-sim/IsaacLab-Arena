# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit-backed live editor for UnresolvedArenaEnvGraphSpec YAMLs.

Wraps :func:`isaaclab_arena.agentic_environment_generation.review_graph.render_html_for_spec`
in a two-pane Streamlit page so the user can edit the ``UnresolvedArenaEnvGraphSpec``
YAML directly in the browser and see the visualization update automatically.

Launch (always via the wrapper in review_graph.py — handles streamlit flags):
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.review_graph \\
        --yaml path/to/spec.yaml

Design:
  * Left pane — ``streamlit-ace`` YAML editor + validation badge + Save button.
    Validation runs on every rerun (i.e. after each editor blur). When the YAML
    is valid and has changed since the last render, the visualization updates
    automatically — no button click required.
  * Right pane — sandboxed iframe with the rendered review HTML.
"""

from __future__ import annotations

import argparse
import traceback
import yaml
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from isaaclab_arena.agentic_environment_generation.review_graph import render_html_for_spec
from isaaclab_arena.environments.arena_env_graph_spec import UnresolvedArenaEnvGraphSpec

# Visualization iframe height. Tuned so the graph + tasks + node grid all
# fit without an outer Streamlit scrollbar swallowing the inner one.
_IFRAME_HEIGHT_PX = 1100


# ---------------------------------------------------------------------------
# Args + session-state init
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Path to the UnresolvedArenaEnvGraphSpec YAML to open in the editor.",
    )
    return parser.parse_args()


@dataclass
class _ValidationResult:
    spec: UnresolvedArenaEnvGraphSpec | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.spec is not None


def _validate_yaml_text(text: str) -> _ValidationResult:
    try:
        raw = yaml.safe_load(text)
        spec = UnresolvedArenaEnvGraphSpec.model_validate(raw)
        return _ValidationResult(spec=spec, error=None)
    except Exception:
        return _ValidationResult(spec=None, error=traceback.format_exc())


def _initialize_state(yaml_path: Path) -> None:
    """Seed ``st.session_state`` from disk exactly once per session.

    We key off ``_yaml_path`` so that if the user passes a different YAML on
    a Streamlit reload (rare — usually the same), we reset cleanly.
    """
    if st.session_state.get("_yaml_path") == str(yaml_path):
        return

    original_text = yaml_path.read_text(encoding="utf-8")

    st.session_state["_yaml_path"] = str(yaml_path)
    st.session_state["original_text"] = original_text
    st.session_state["edited_text"] = original_text
    # The text whose render is currently displayed. Starts == original so the
    # first paint shows the on-disk file (and "Regenerate" is correctly
    # disabled until the user edits something).
    st.session_state["last_rendered_text"] = original_text
    st.session_state["save_path"] = str(yaml_path)

    initial = _validate_yaml_text(original_text)
    if not initial.is_valid:
        # Defensive: if the on-disk file is already broken we still want to
        # show *something*, but we won't pre-render it. The user fixes the
        # YAML in the editor, then hits Regenerate.
        st.session_state["rendered_html"] = _BROKEN_PLACEHOLDER_HTML
    else:
        st.session_state["rendered_html"] = render_html_for_spec(initial.spec)


# Tiny standalone HTML used when the on-disk YAML is itself invalid.
_BROKEN_PLACEHOLDER_HTML = """<!DOCTYPE html><html><body style="
    font-family: ui-monospace, monospace;
    background:#15181d; color:#e4e6eb; padding:24px; margin:0;">
<p>No visualization yet — fix the YAML errors to auto-render.</p>
</body></html>"""


# ---------------------------------------------------------------------------
# UI panels
# ---------------------------------------------------------------------------


def _render_validation_badge(validation: _ValidationResult) -> None:
    """Show a green tick + summary, or a red cross + the raw exception text."""
    if validation.is_valid:
        spec = validation.spec
        st.success(
            f"Valid spec — {spec.env_name} · {len(spec.nodes)} nodes · "
            f"{len(spec.tasks)} tasks · initial state: {spec.initial_state_spec.id}",
            icon="✅",
        )
    else:
        st.error(f"Invalid YAML\n\n```\n{validation.error}\n```", icon="🛑")


def _render_save_button(validation: _ValidationResult) -> None:
    """Render the Save button. Disabled while the YAML is invalid."""
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
            # Update "original" so future comparisons are against the saved file.
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


def _render_editor_panel(yaml_path: Path) -> _ValidationResult:
    """Left pane. Returns the validation result for the current editor text.

    Returning the validation result (rather than stashing it in session_state)
    keeps the data flow inside one render pass and avoids a stale-state class
    of bug where the badge and the buttons disagree.
    """
    # Lazy import so the module is importable from environments that don't
    # have streamlit-ace installed yet (we surface a clean error message
    # rather than ImportError at module load).
    try:
        from streamlit_ace import st_ace  # noqa: PLC0415
    except ImportError as exc:
        # See review_graph._serve_live_editor for why --user --ignore-installed
        # is required inside the isaaclab_arena container.
        st.error(
            "`streamlit-ace` is not installed. Inside the isaaclab_arena container run:\n"
            "`python -m pip install --user --ignore-installed streamlit-ace`\n\n"
            f"Underlying error: {exc}",
            icon="🛑",
        )
        st.stop()

    st.subheader("YAML editor")
    st.caption(f"Source: `{yaml_path}`")

    # ``auto_update=False`` commits on blur / Ctrl+Enter rather than on every
    # keystroke, showing an "Apply" button in the editor toolbar.
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
        # Key bound to the YAML path so swapping --yaml between sessions
        # forces ace to remount with the new content.
        key=f"ace_editor::{yaml_path}",
    )
    if new_text is not None:
        st.session_state["edited_text"] = new_text

    validation = _validate_yaml_text(st.session_state["edited_text"])
    _render_validation_badge(validation)

    # Auto-render whenever the YAML is valid and has changed since the last
    # render. This runs before the right pane is drawn, so the updated HTML
    # is already in session_state when the iframe is mounted — no extra rerun
    # needed.
    edited_since_render = st.session_state["edited_text"] != st.session_state["last_rendered_text"]
    if validation.is_valid and edited_since_render:
        with st.spinner("Rendering visualization…"):
            st.session_state["rendered_html"] = render_html_for_spec(validation.spec)
        st.session_state["last_rendered_text"] = st.session_state["edited_text"]
        st.toast("Visualization updated.", icon="🔄")

    _render_save_button(validation)
    return validation


def _render_visualization_panel() -> None:
    """Right pane — iframe-mount the cached rendered HTML."""
    st.subheader("Visualization")
    st.caption("Updates automatically when the YAML is valid.")

    # ``st.components.v1.html`` wraps the payload in a sandboxed iframe, which
    # is what we want — the mermaid CDN script and the static CSS stay
    # isolated from Streamlit's own DOM.
    st.components.v1.html(
        st.session_state["rendered_html"],
        height=_IFRAME_HEIGHT_PX,
        scrolling=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="UnresolvedArenaEnvGraphSpec live editor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    args = _parse_args()
    yaml_path = args.yaml.resolve()
    if not yaml_path.exists():
        st.error(f"YAML file not found: {yaml_path}", icon="🛑")
        st.stop()

    _initialize_state(yaml_path)

    st.markdown("### UnresolvedArenaEnvGraphSpec live editor")
    left, right = st.columns([2, 3], gap="large")
    with left:
        _render_editor_panel(yaml_path)
    with right:
        _render_visualization_panel()


# Streamlit invokes the script top-level on every rerun, so we run main()
# unconditionally. The standard ``if __name__ == "__main__"`` guard would
# also work under ``streamlit run`` but is unnecessary — this module is only
# ever loaded as the Streamlit entrypoint.
main()
