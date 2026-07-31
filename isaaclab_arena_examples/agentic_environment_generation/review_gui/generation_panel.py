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
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadySourceKind,
    simready_search_config_from_cli,
)
from isaaclab_arena.assets.simready_object_library import DEFAULT_SIMREADY_SERVICE_URL
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


def _simready_config_from_session() -> tuple[bool, object]:
    """Read SimReady GUI settings from Streamlit session state."""
    enabled = bool(st.session_state.get("enable_simready_search", False))
    config = simready_search_config_from_cli(
        enabled=enabled,
        source=st.session_state.get("simready_source", SimReadySourceKind.ISAAC_SIM_GA.value),
        s3_url=st.session_state.get("simready_s3_url") or None,
        service_url=st.session_state.get("simready_service_url") or None,
        project_config_path=st.session_state.get("simready_project_config") or None,
        indexed_path=st.session_state.get("simready_indexed_dir") or None,
        max_results_per_object=int(st.session_state.get("simready_max_results_per_object", 1)),
    )
    return enabled, config


def _get_generation_agent() -> EnvironmentGenerationAgent | None:
    """Lazy-init the LLM agent when ``NV_API_KEY`` is available."""
    if st.session_state.get("generation_agent_error"):
        return None
    simready_enabled, simready_config = _simready_config_from_session()
    agent_key = (
        "generation_agent",
        simready_enabled,
        simready_config.source.value,
        simready_config.s3_url,
        simready_config.service_url,
        simready_config.project_config_path,
        simready_config.indexed_path,
        simready_config.max_results_per_object,
    )
    agent = st.session_state.get(agent_key)
    if agent is not None:
        return agent
    try:
        agent = EnvironmentGenerationAgent(
            enable_simready_search=simready_enabled,
            simready_config=simready_config,
        )
    except AssertionError as exc:
        st.session_state["generation_agent_error"] = str(exc)
        return None
    except Exception as exc:
        st.session_state["generation_agent_error"] = f"{type(exc).__name__}: {exc}"
        return None
    st.session_state[agent_key] = agent
    st.session_state.pop("generation_agent_error", None)
    return agent


def _apply_generated_yaml(
    yaml_text: str,
    *,
    spec: ArenaEnvGraphSpec | None = None,
    validation_error: str | None = None,
) -> None:
    """Push generated spec YAML into the editor; the visualization panel refreshes in the viz fragment."""
    st.session_state["edited_text"] = yaml_text
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["last_rendered_text"] = ""
    st.session_state["rendered_visualization"] = None
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


def _finish(severity: str, message: str) -> tuple[bool, str]:
    """Record the banner severity for the generate button and return its ``(ok, message)`` pair.

    The severity is decided here, where the outcome is known, rather than read back out of the
    message text — a reworded message must not silently turn a failure green.

    Args:
        severity: Streamlit banner level, ``"success"`` or ``"warning"``.
        message: Text shown in the banner.
    """
    st.session_state["_generation_severity"] = severity
    return True, message


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
        simready_enabled, _ = _simready_config_from_session()
        spec, data = agent.generate_spec(
            prompt,
            asset_catalog=catalogues.asset_catalogue,
            relation_catalog=catalogues.relation_catalogue,
            task_catalog=catalogues.task_catalogue,
            enable_simready_search=simready_enabled,
        )
    except Exception:
        return False, traceback.format_exc()

    try:
        yaml_text = yaml.safe_dump(
            spec.to_dict() if spec is not None else (data or {}),
            sort_keys=False,
        )
    except Exception:
        return False, traceback.format_exc()

    if spec is None:
        traces = "\n".join(agent.traces) or "unknown validation error"
        if agent.unavailable_objects:
            headline = (
                f"No asset is available for: {', '.join(agent.unavailable_objects)}. "
                "Rephrase the prompt with a more common object, or register the asset in Arena."
            )
        else:
            headline = "Agent returned an invalid spec."
        _apply_generated_yaml(yaml_text, validation_error=f"{headline}\n{traces}")
        return _finish("warning", f"{headline}\nLoaded into the YAML editor.\n{traces}")

    _apply_generated_yaml(yaml_text, spec=spec)

    out_dir = Path(st.session_state["out_dir"])
    path, error = try_save_env_graph_spec(spec, out_dir)
    trace_suffix = ""
    if simready_enabled and agent.traces:
        trace_suffix = "\n\nGeneration traces:\n" + "\n".join(agent.traces)
    if error is not None:
        return _finish(
            "warning", f"Spec generated and loaded into the YAML editor, but save failed: {error}{trace_suffix}"
        )

    st.session_state["save_path"] = str(path)
    return _finish("success", f"Spec generated, loaded into the YAML editor, and saved to {path}.{trace_suffix}")


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

    with st.expander("SimReady search", expanded=bool(st.session_state.get("enable_simready_search", False))):
        st.session_state["enable_simready_search"] = st.checkbox(
            "Enable SimReady search",
            value=bool(st.session_state.get("enable_simready_search", False)),
            help="Search Isaac Sim GA SimReady props for objects the Arena asset catalog does not cover.",
        )
        source_options = [kind.value for kind in SimReadySourceKind]
        st.session_state["simready_source"] = st.selectbox(
            "SimReady source",
            options=source_options,
            index=source_options.index(st.session_state.get("simready_source", SimReadySourceKind.ISAAC_SIM_GA.value)),
        )
        st.session_state["simready_max_results_per_object"] = st.number_input(
            "Max results per object",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("simready_max_results_per_object", 1)),
        )
        # Each source reads exactly one location, so only that one is offered. Showing all four
        # invites filling in a field the search then ignores.
        source = st.session_state["simready_source"]
        if source == SimReadySourceKind.S3.value:
            st.session_state["simready_s3_url"] = st.text_input(
                "S3 URL",
                value=st.session_state.get("simready_s3_url", ""),
                placeholder="Leave empty for the Isaac Sim 6.0 GA SimReady bucket",
            )
        elif source == SimReadySourceKind.SERVICE.value:
            st.session_state["simready_service_url"] = st.text_input(
                "Service URL",
                value=st.session_state.get("simready_service_url", ""),
                placeholder=DEFAULT_SIMREADY_SERVICE_URL,
            )
        elif source == SimReadySourceKind.CACHE.value:
            st.session_state["simready_project_config"] = st.text_input(
                "project_config.toml path",
                value=st.session_state.get("simready_project_config", ""),
                help="Manifest describing the layout of an already-cached SimReady workspace.",
            )
        elif source == SimReadySourceKind.INDEXED.value:
            st.session_state["simready_indexed_dir"] = st.text_input(
                "Indexed directory or S3 prefix",
                value=st.session_state.get("simready_indexed_dir", ""),
            )
        else:
            st.caption("Searches the Isaac Sim 6.0 GA SimReady library — no further configuration needed.")

    if st.button("Generate spec", type="primary", width="stretch"):
        with st.spinner("Generating spec (LLM call)…"):
            ok, message = run_generation_pipeline(st.session_state["generation_prompt"])
        if ok:
            severity = st.session_state.pop("_generation_severity", "success")
            st.session_state["_generation_feedback"] = (severity, message)
            st.rerun()
        else:
            st.error(f"Generation failed\n\n```\n{message}\n```", icon="🛑")
