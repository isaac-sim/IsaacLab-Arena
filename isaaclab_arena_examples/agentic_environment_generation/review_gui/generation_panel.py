# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (
    SpecParseResult,
    try_save_initial_graph_spec,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.sidecar_service import (
    render_dashboard_with_thumbnails,
)

DEFAULT_GENERATION_PROMPT = (
    "franka pick up avocado from the maple table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)


@dataclass
class CatalogueBundle:
    """Asset/relation/task vocabulary for the env-generation agent."""

    asset_catalogue: AssetCatalogue
    relation_catalogue: RelationCatalogue
    task_catalogue: TaskCatalogue


@st.cache_resource(show_spinner="Building asset catalogues (first run)…")
def get_catalogue_bundle() -> CatalogueBundle:
    """Build and cache registry-backed catalogues for LLM prompt assembly."""
    return CatalogueBundle(
        asset_catalogue=build_asset_catalogue(),
        relation_catalogue=build_relation_catalogue(),
        task_catalogue=build_task_catalogue(),
    )


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
    error_stages = ASSET_ERROR_STAGES | IntentCompiler.INTENT_ERROR_STAGES
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


def _apply_generated_yaml(yaml_text: str, *, spec: ArenaEnvInitialGraphSpec | None = None) -> None:
    """Push compiled spec YAML into the editor and sync the dashboard preview."""
    st.session_state["edited_text"] = yaml_text
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["last_rendered_text"] = yaml_text
    if spec is not None:
        st.session_state["rendered_html"] = render_dashboard_with_thumbnails(spec)
        st.session_state["_validation_text"] = yaml_text
        st.session_state["_validation_result"] = SpecParseResult(spec=spec, error=None)
    else:
        st.session_state["rendered_html"] = ""
        st.session_state.pop("_validation_text", None)
        st.session_state.pop("_validation_result", None)


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
        catalogues = get_catalogue_bundle()
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

    _apply_generated_yaml(yaml_text, spec=spec)

    if reasoning:
        st.session_state["last_generation_reasoning"] = reasoning
    if trace:
        st.session_state["last_generation_trace"] = trace

    out_dir = Path(st.session_state["out_dir"])
    paths, error = try_save_initial_graph_spec(spec, out_dir)
    if error is not None:
        if has_resolution_errors:
            error_trace = _format_trace_lines(trace, errors_only=True)
            return (
                True,
                (
                    "Spec generated with resolution warnings — review the trace below and edit the YAML as needed.\n\n"
                    f"{error}\n\n{error_trace}"
                ),
            )
        return True, f"Spec generated and loaded into the YAML editor, but save failed: {error}"

    initial_path, linked_path = paths
    st.session_state["save_path"] = str(initial_path)

    saved_msg = f"Wrote {initial_path} and {linked_path}."

    if has_resolution_errors:
        error_trace = _format_trace_lines(trace, errors_only=True)
        return (
            True,
            (
                "Spec generated with resolution warnings — review the trace below and edit the YAML as needed.\n\n"
                f"{saved_msg}\n\n{error_trace}"
            ),
        )

    return True, f"Spec generated, loaded into the YAML editor, and saved.\n\n{saved_msg}"


def render_generation_panel() -> None:
    """Prompt input and generate-spec controls (top of the left column)."""
    st.subheader("Generate from prompt")
    st.caption("Calls the env-generation agent (LLM) then compiles intent in-process.")

    prompt = st.text_area(
        "Prompt",
        value=st.session_state.get("generation_prompt", DEFAULT_GENERATION_PROMPT),
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
            if "resolution warnings" in message or "save failed" in message.lower():
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
