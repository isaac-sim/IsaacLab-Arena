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

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    AssetCatalogue,
    EnvironmentGenerationAgent,
    RelationCatalogue,
    TaskCatalogue,
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (
    SpecParseResult,
    try_save_env_graph_spec,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.visualization_panel import reset_viz_render_state

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


def _apply_generated_yaml(
    yaml_text: str,
    *,
    spec: ArenaEnvGraphSpec | None = None,
    validation_error: str | None = None,
) -> None:
    """Push generated spec YAML into the editor; dashboard preview refreshes in the viz fragment."""
    st.session_state["edited_text"] = yaml_text
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["last_rendered_text"] = ""
    st.session_state["rendered_html"] = ""
    reset_viz_render_state()
    if spec is not None:
        st.session_state["_validation_text"] = yaml_text
        st.session_state["_validation_result"] = SpecParseResult(spec=spec, error=None)
        st.session_state["_defer_viz_render"] = True
    elif validation_error is not None:
        st.session_state["_validation_text"] = yaml_text
        st.session_state["_validation_result"] = SpecParseResult(spec=None, error=validation_error)
    else:
        st.session_state.pop("_validation_text", None)
        st.session_state.pop("_validation_result", None)


def run_generation_pipeline(prompt: str) -> tuple[bool, str]:
    """Call the LLM and load the returned environment graph spec YAML into the editor."""
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
        spec, data = agent.generate_spec(
            prompt,
            asset_catalog=catalogues.asset_catalogue,
            relation_catalog=catalogues.relation_catalogue,
            task_catalog=catalogues.task_catalogue,
        )
    except Exception:
        return False, traceback.format_exc()

    try:
        yaml_text = yaml.safe_dump(
            spec.to_dict() if spec is not None else data,
            sort_keys=False,
        )
    except Exception:
        return False, traceback.format_exc()

    if spec is None:
        traces = "\n".join(agent.last_validation_traces) or "unknown validation error"
        error = f"Agent returned an invalid spec:\n{traces}"
        _apply_generated_yaml(yaml_text, validation_error=error)
        return True, f"Invalid spec loaded into the YAML editor.\n{traces}"

    _apply_generated_yaml(yaml_text, spec=spec)

    out_dir = Path(st.session_state["out_dir"])
    path, error = try_save_env_graph_spec(spec, out_dir)
    if error is not None:
        return True, f"Spec generated and loaded into the YAML editor, but save failed: {error}"

    st.session_state["save_path"] = str(path)
    return True, f"Spec generated, loaded into the YAML editor, and saved to {path}."


def render_generation_panel() -> None:
    """Prompt input and generate-spec controls (top of the left column)."""
    st.subheader("Generate from prompt")
    st.caption("Calls the env-generation agent (LLM) and loads the returned environment graph spec.")

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

    if st.button("Generate spec", type="primary", use_container_width=True):
        with st.spinner("Generating spec (LLM call)…"):
            ok, message = run_generation_pipeline(st.session_state["generation_prompt"])
        if ok:
            lowered = message.lower()
            if "invalid spec" in lowered or "save failed" in lowered:
                st.session_state["_generation_feedback"] = ("warning", message)
            else:
                st.session_state["_generation_feedback"] = ("success", message)
            st.rerun()
        else:
            st.error(f"Generation failed\n\n```\n{message}\n```", icon="🛑")
