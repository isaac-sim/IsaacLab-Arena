# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit UI for the initial-graph live editor.

Launch via :mod:`~isaaclab_arena_examples.agentic_environment_generation.gui_runner`:

    # Default — prompt-only (empty editor until you generate or paste YAML):
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.gui_runner

    # Open an existing spec:
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.gui_runner \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml

Natural-language generation calls the LLM from Streamlit (``NV_API_KEY``) and
compiles the returned intent in-process with :class:`IntentCompiler`.
"""

from __future__ import annotations

import argparse
import traceback
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from isaaclab_arena.agentic_environment_generation.asset_matcher import ASSET_ERROR_STAGES
from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    AssetCatalogue,
    EnvironmentGenerationAgent,
    RelationCatalogue,
    TaskCatalogue,
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html

_IFRAME_HEIGHT_PX = 1100

_DEFAULT_GENERATION_PROMPT = (
    "franka pick up avocado from the maple table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)

_BROKEN_PLACEHOLDER_HTML = """<!DOCTYPE html><html><body style="
    font-family: ui-monospace, monospace;
    background:#15181d; color:#e4e6eb; padding:24px; margin:0;">
<p>No visualization yet — fix the YAML errors to auto-render.</p>
</body></html>"""

_DEFAULT_SAVE_PATH = "isaaclab_arena_environments/llm_generated/generated_spec.yaml"


@dataclass
class ValidationResult:
    """Outcome of parsing and validating YAML text as an initial graph spec."""

    spec: ArenaEnvInitialGraphSpec | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        """True when ``spec`` parsed successfully."""
        return self.spec is not None


@dataclass
class CatalogueBundle:
    """Asset/relation/task vocabulary for the env-generation agent."""

    asset_catalogue: AssetCatalogue
    relation_catalogue: RelationCatalogue
    task_catalogue: TaskCatalogue


@st.cache_resource(show_spinner="Building asset catalogues (first run)…")
def _get_catalogue_bundle() -> CatalogueBundle:
    """Build and cache registry-backed catalogues for LLM prompt assembly."""
    return CatalogueBundle(
        asset_catalogue=build_asset_catalogue(),
        relation_catalogue=build_relation_catalogue(),
        task_catalogue=build_task_catalogue(),
    )


def validate_yaml_text(text: str) -> ValidationResult:
    """Parse ``text`` as YAML and validate it as an :class:`ArenaEnvInitialGraphSpec`."""
    if not text.strip():
        return ValidationResult(spec=None, error=None)

    try:
        raw = yaml.safe_load(text)
    except Exception:
        return ValidationResult(spec=None, error=traceback.format_exc())

    if raw is None:
        return ValidationResult(spec=None, error="YAML is empty")
    if not isinstance(raw, dict):
        return ValidationResult(spec=None, error=f"Expected mapping, got {type(raw).__name__}")

    try:
        spec = ArenaEnvInitialGraphSpec.model_validate(raw)
    except Exception:
        return ValidationResult(spec=None, error=traceback.format_exc())

    return ValidationResult(spec=spec, error=None)


def _get_generation_agent() -> EnvironmentGenerationAgent | None:
    """Lazy-init the LLM agent when ``NV_API_KEY`` is available."""
    if st.session_state.get("generation_agent_error"):
        return None
    agent = st.session_state.get("generation_agent")
    if agent is not None:
        return agent
    try:
        agent = EnvironmentGenerationAgent()
    except AssertionError as exc:
        st.session_state["generation_agent_error"] = str(exc)
        return None
    except Exception as exc:
        st.session_state["generation_agent_error"] = f"{type(exc).__name__}: {exc}"
        return None
    st.session_state["generation_agent"] = agent
    st.session_state.pop("generation_agent_error", None)
    return agent


def _format_trace_lines(trace: list[dict[str, Any]], *, errors_only: bool = False) -> str:
    """Format intent-compiler trace events as fixed-width log lines."""
    error_stages = ASSET_ERROR_STAGES | IntentCompiler._ERROR_TRACE_STAGES
    lines: list[str] = []
    for event in trace:
        stage = event.get("stage", "")
        if errors_only and stage not in error_stages:
            continue
        chosen = event.get("chosen")
        chosen_str = chosen if chosen is not None else "<none>"
        note = event.get("note") or ""
        note_str = f"  [{note}]" if note else ""
        lines.append(f"{stage:34s} {event.get('query', ''):24s} -> {chosen_str}{note_str}")
    return "\n".join(lines)


def _apply_generated_yaml(yaml_text: str) -> None:
    """Push compiled spec YAML into the editor and force a re-render on the next pass."""
    st.session_state["edited_text"] = yaml_text
    st.session_state["last_rendered_text"] = ""
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def run_generation_pipeline(prompt: str) -> tuple[bool, str]:
    """Call the LLM, compile intent in-process, and load YAML into the editor."""
    prompt = prompt.strip()
    if not prompt:
        return False, "Enter a prompt describing the environment."

    agent = _get_generation_agent()
    if agent is None:
        err = st.session_state.get(
            "generation_agent_error",
            "Set NV_API_KEY in the environment before generating specs.",
        )
        return False, err

    try:
        catalogues = _get_catalogue_bundle()
    except Exception:
        return False, traceback.format_exc()

    try:
        intent, _raw = agent.generate_spec(
            prompt,
            asset_catalog=catalogues.asset_catalogue,
            relation_catalog=catalogues.relation_catalogue,
            task_catalog=catalogues.task_catalogue,
        )
    except Exception:
        return False, traceback.format_exc()

    try:
        compiler = IntentCompiler()
        spec = compiler.compile(intent)
        yaml_text = yaml.safe_dump(spec.to_dict(), sort_keys=False)
        trace = [asdict(event) for event in compiler.trace]
        has_resolution_errors = compiler.has_resolution_errors
        reasoning = intent.reasoning
    except Exception:
        return False, traceback.format_exc()

    _apply_generated_yaml(yaml_text)

    if reasoning:
        st.session_state["last_generation_reasoning"] = reasoning
    if trace:
        st.session_state["last_generation_trace"] = trace

    if has_resolution_errors:
        error_trace = _format_trace_lines(trace, errors_only=True)
        return (
            True,
            (
                "Spec generated with resolution warnings — review the trace below and edit the YAML as needed.\n\n"
                f"{error_trace}"
            ),
        )

    return True, "Spec generated and loaded into the YAML editor."


def parse_args() -> argparse.Namespace:
    """Parse Streamlit CLI args forwarded after ``--`` by :mod:`gui_runner`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Optional path to an ArenaEnvInitialGraphSpec YAML to open in the editor.",
    )
    return parser.parse_args()


def initialize_state(yaml_path: Path | None) -> None:
    """Seed ``st.session_state`` from disk exactly once per session."""
    session_key = str(yaml_path.resolve()) if yaml_path is not None else ""
    if st.session_state.get("_yaml_path") == session_key:
        return

    st.session_state["_yaml_path"] = session_key
    st.session_state.setdefault("generation_prompt", _DEFAULT_GENERATION_PROMPT)
    st.session_state.setdefault("editor_version", 0)

    if yaml_path is None:
        st.session_state["original_text"] = ""
        st.session_state["edited_text"] = ""
        st.session_state["last_rendered_text"] = ""
        st.session_state["rendered_html"] = ""
        st.session_state["save_path"] = _DEFAULT_SAVE_PATH
        return

    original_text = yaml_path.read_text(encoding="utf-8")

    st.session_state["original_text"] = original_text
    st.session_state["edited_text"] = original_text
    st.session_state["last_rendered_text"] = original_text
    st.session_state["save_path"] = str(yaml_path)

    initial = validate_yaml_text(original_text)
    if not initial.is_valid:
        st.session_state["rendered_html"] = _BROKEN_PLACEHOLDER_HTML
    else:
        st.session_state["rendered_html"] = render_dashboard_html(initial.spec)


def render_validation_badge(validation: ValidationResult) -> None:
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


def render_save_button(validation: ValidationResult) -> None:
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
            help="Defaults to the YAML file passed via --yaml, or a generated-spec path when none was given.",
        )
        if new_path and new_path != save_path_str:
            st.session_state["save_path"] = new_path


def render_editor_panel(yaml_path: Path | None) -> ValidationResult:
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
    if validation.is_valid and edited_since_render:
        with st.spinner("Rendering visualization…"):
            st.session_state["rendered_html"] = render_dashboard_html(validation.spec)
        st.session_state["last_rendered_text"] = st.session_state["edited_text"]
        st.toast("Visualization updated.", icon="🔄")

    render_save_button(validation)
    return validation


def render_generation_panel() -> None:
    """Prompt input and generate-spec controls (top of the left column)."""
    st.subheader("Generate from prompt")
    st.caption("Calls the env-generation agent (LLM) then compiles intent in-process.")

    prompt = st.text_area(
        "Prompt",
        value=st.session_state.get("generation_prompt", _DEFAULT_GENERATION_PROMPT),
        height=120,
        placeholder="Describe the robot task, scene, objects, and distractors…",
    )
    st.session_state["generation_prompt"] = prompt

    agent_error = st.session_state.get("generation_agent_error")
    if agent_error:
        st.info(f"LLM agent unavailable: {agent_error}", icon="ℹ️")

    if st.button("Generate & compile", type="primary", use_container_width=True):
        with st.spinner("Generating spec (LLM call + intent compile)…"):
            ok, message = run_generation_pipeline(st.session_state["generation_prompt"])
        if ok:
            if "resolution warnings" in message:
                st.warning(message, icon="⚠️")
            else:
                st.success(message, icon="✅")
            st.rerun()
        else:
            st.error(f"Generation failed\n\n```\n{message}\n```", icon="🛑")

    reasoning = st.session_state.get("last_generation_reasoning")
    if reasoning:
        with st.expander("Agent reasoning (last run)", expanded=False):
            st.markdown(reasoning)

    trace = st.session_state.get("last_generation_trace")
    if trace:
        with st.expander("Resolution trace (last run)", expanded=False):
            st.code(_format_trace_lines(trace), language=None)


def render_visualization_panel() -> None:
    """Embed the rendered dashboard HTML in the right-hand column."""
    st.subheader("Visualization")
    if not st.session_state.get("last_rendered_text", "").strip():
        st.caption("Generate or enter valid YAML to see the visualization.")
        return

    st.caption("Updates automatically when the YAML is valid.")
    st.components.v1.html(
        st.session_state["rendered_html"],
        height=_IFRAME_HEIGHT_PX,
        scrolling=True,
    )


def main() -> None:
    """Build the two-column Streamlit layout for generation, editing, and preview."""
    st.set_page_config(
        page_title="ArenaEnvInitialGraphSpec live editor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    args = parse_args()
    yaml_path = args.yaml.resolve() if args.yaml is not None else None
    if yaml_path is not None and not yaml_path.exists():
        st.error(f"YAML file not found: {yaml_path}", icon="🛑")
        st.stop()

    initialize_state(yaml_path)

    st.markdown("### ArenaEnvInitialGraphSpec live editor")
    left, right = st.columns([2, 3], gap="large")
    with left:
        render_generation_panel()
        render_editor_panel(yaml_path)
    with right:
        render_visualization_panel()


main()
