# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit UI for the initial-graph live editor.

Launch via :mod:`~isaaclab_arena_examples.agentic_environment_generation.review_gui.server`:

    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.review_gui.server \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml

    # Prompt-only (empty editor until you generate or paste YAML):
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.review_gui.server

Registry lookups (task kinds, relation kinds, asset USD paths) run in a
persistent SimApp sidecar so YAML re-validation stays fast after the first
~30s sidecar boot. Thumbnails are rendered live by the same sidecar.

Natural-language generation calls the LLM from Streamlit (``NV_API_KEY``) and
compiles the returned intent in the sidecar where registries are warm.
"""

from __future__ import annotations

import argparse
import atexit
import traceback
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from isaaclab_arena.agentic_environment_generation.asset_matcher import ASSET_ERROR_STAGES
from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    AssetCatalogue,
    EnvironmentGenerationAgent,
    RelationCatalogue,
    RelationCatalogueEntry,
    TaskCatalogue,
    TaskCatalogueEntry,
)
from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.dashboard import render_dashboard_html
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar_client import (
    SimAppSidecar,
    SimAppSidecarError,
)

_IFRAME_HEIGHT_PX = 1100

_SKIP_REGISTRY_CONTEXT: dict[str, Any] = {"skip_registry": True}

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


@st.cache_resource(show_spinner="Booting Isaac Sim sidecar (≈30s first run, cached afterwards)…")
def _get_simapp_sidecar() -> SimAppSidecar | None:
    """Spawn the SimApp sidecar once per Streamlit server process."""
    sidecar = SimAppSidecar()
    try:
        sidecar.start()
    except SimAppSidecarError as exc:
        print(f"[review_gui] SimApp sidecar failed to start: {exc}", flush=True)
        return None

    atexit.register(sidecar.close)
    return sidecar


def _ensure_sidecar() -> SimAppSidecar | None:
    """Return a healthy sidecar, re-spawning if the cached one died."""
    sidecar = _get_simapp_sidecar()
    if sidecar is not None and sidecar.is_alive():
        return sidecar
    if sidecar is not None:
        sidecar.close()
    _get_simapp_sidecar.clear()
    return _get_simapp_sidecar()


def _spec_from_sidecar_dict(spec_dict: dict[str, Any]) -> ArenaEnvInitialGraphSpec:
    """Rebuild a validated spec locally without registry imports."""
    return ArenaEnvInitialGraphSpec.model_validate(spec_dict, context=_SKIP_REGISTRY_CONTEXT)


def _render_with_thumbnails(spec: ArenaEnvInitialGraphSpec) -> str:
    """Render review HTML, asking the sidecar for live USD thumbnails."""
    sidecar = _ensure_sidecar()
    if sidecar is None:
        st.warning(
            "Isaac Sim sidecar is unavailable — showing placeholder thumbnails. "
            "Check the terminal where you launched the server for the underlying error.",
            icon="⚠️",
        )
        return render_dashboard_html(spec)

    try:
        thumbnails = sidecar.render_spec(spec)
    except SimAppSidecarError as exc:
        st.error(
            f"Sidecar render failed; showing placeholder thumbnails.\n\n```\n{exc}\n```",
            icon="🛑",
        )
        with st.spinner("Resetting the SimApp sidecar…"):
            _get_simapp_sidecar.clear()
        return render_dashboard_html(spec)

    return render_dashboard_html(spec, thumbnails=thumbnails if thumbnails else None)


@dataclass
class ValidationResult:
    """Outcome of parsing and validating YAML text as an initial graph spec."""

    spec: ArenaEnvInitialGraphSpec | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.spec is not None


def validate_yaml_text(text: str) -> ValidationResult:
    """Parse YAML and validate via the SimApp sidecar (registry lookups run there)."""
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

    sidecar = _ensure_sidecar()
    if sidecar is None:
        return ValidationResult(
            spec=None,
            error=(
                "SimApp sidecar is unavailable — cannot validate registry entries. "
                "Check the terminal where you launched the server."
            ),
        )

    try:
        response = sidecar.validate_yaml_text(text)
    except SimAppSidecarError as exc:
        _get_simapp_sidecar.clear()
        return ValidationResult(spec=None, error=str(exc))

    if not response.get("ok"):
        err = response.get("error", "validation failed")
        tb = response.get("traceback", "")
        message = f"{err}\n\n{tb}" if tb else str(err)
        return ValidationResult(spec=None, error=message)

    try:
        spec = _spec_from_sidecar_dict(response["spec_dict"])
    except Exception:
        return ValidationResult(spec=None, error=traceback.format_exc())

    return ValidationResult(spec=spec, error=None)


@dataclass
class CatalogueBundle:
    """Asset/relation/task vocabulary fetched from the SimApp sidecar."""

    asset_catalogue: AssetCatalogue
    relation_catalogue: RelationCatalogue
    task_catalogue: TaskCatalogue


def _catalogues_from_sidecar_response(response: dict[str, Any]) -> CatalogueBundle:
    asset_raw = response["asset_catalogue"]
    relation_raw = response["relation_catalogue"]
    task_raw = response["task_catalogue"]
    return CatalogueBundle(
        asset_catalogue=AssetCatalogue(
            embodiments=list(asset_raw["embodiments"]),
            backgrounds=list(asset_raw["backgrounds"]),
            objects=list(asset_raw["objects"]),
        ),
        relation_catalogue=RelationCatalogue(
            relations=[RelationCatalogueEntry(**entry) for entry in relation_raw["relations"]],
        ),
        task_catalogue=TaskCatalogue(
            tasks=[TaskCatalogueEntry(**entry) for entry in task_raw["tasks"]],
        ),
    )


def _get_catalogue_bundle() -> CatalogueBundle | None:
    """Return cached catalogues built in the sidecar (registry lookups run there)."""
    cached = st.session_state.get("catalogue_bundle")
    if cached is not None:
        return cached

    sidecar = _ensure_sidecar()
    if sidecar is None:
        return None

    try:
        response = sidecar.build_catalogues()
    except SimAppSidecarError as exc:
        st.session_state["catalogue_error"] = str(exc)
        _get_simapp_sidecar.clear()
        return None

    if not response.get("ok"):
        st.session_state["catalogue_error"] = response.get("error", "catalogue build failed")
        return None

    bundle = _catalogues_from_sidecar_response(response)
    st.session_state["catalogue_bundle"] = bundle
    st.session_state.pop("catalogue_error", None)
    return bundle


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
    """Call the LLM, compile intent in the sidecar, and load YAML into the editor."""
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

    catalogues = _get_catalogue_bundle()
    if catalogues is None:
        err = st.session_state.get(
            "catalogue_error",
            "SimApp sidecar is unavailable — cannot build asset catalogues.",
        )
        return False, err

    try:
        intent_data, _raw = agent.fetch_intent_from_prompt(
            prompt,
            asset_catalog=catalogues.asset_catalogue,
            relation_catalog=catalogues.relation_catalogue,
            task_catalog=catalogues.task_catalogue,
        )
    except Exception:
        return False, traceback.format_exc()

    sidecar = _ensure_sidecar()
    if sidecar is None:
        return False, "SimApp sidecar is unavailable — cannot compile intent."

    try:
        response = sidecar.compile_intent(intent_data)
    except SimAppSidecarError as exc:
        _get_simapp_sidecar.clear()
        return False, str(exc)

    if not response.get("ok"):
        err = response.get("error", "intent compile failed")
        tb = response.get("traceback", "")
        message = f"{err}\n\n{tb}" if tb else str(err)
        return False, message

    try:
        yaml_text = yaml.safe_dump(response["spec_dict"], sort_keys=False)
    except Exception:
        return False, traceback.format_exc()

    _apply_generated_yaml(yaml_text)

    reasoning = response.get("reasoning", "")
    if reasoning:
        st.session_state["last_generation_reasoning"] = reasoning

    trace = response.get("trace") or []
    if trace:
        st.session_state["last_generation_trace"] = trace

    if response.get("has_resolution_errors"):
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
        st.session_state["rendered_html"] = _render_with_thumbnails(initial.spec)


def render_validation_badge(validation: ValidationResult) -> None:
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
            st.session_state["rendered_html"] = _render_with_thumbnails(validation.spec)
        st.session_state["last_rendered_text"] = st.session_state["edited_text"]
        st.toast("Visualization updated.", icon="🔄")

    render_save_button(validation)
    return validation


def render_generation_panel() -> None:
    """Prompt input and generate-spec controls (top of the left column)."""
    st.subheader("Generate from prompt")
    st.caption("Calls the env-generation agent (LLM) then compiles intent in the SimApp sidecar.")

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
        with st.spinner("Generating spec (LLM call + sidecar compile)…"):
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
