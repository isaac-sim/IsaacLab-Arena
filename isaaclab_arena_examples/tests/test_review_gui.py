# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from isaaclab_arena.agentic_environment_generation.asset_matcher import ASSET_ERROR_STAGES
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (
    DEFAULT_SAVE_PATH,
    SpecParseResult,
    validate_yaml_text,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel import (
    DEFAULT_GENERATION_PROMPT,
    _apply_generated_yaml,
    _format_trace_lines,
    run_generation_pipeline,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.streamlit_ui import initialize_state, parse_args

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_SPEC_YAML_PATH = _REPO_ROOT / "isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml"


@pytest.fixture
def valid_spec_yaml() -> str:
    return _VALID_SPEC_YAML_PATH.read_text(encoding="utf-8")


@pytest.fixture
def valid_spec(valid_spec_yaml: str) -> ArenaEnvInitialGraphSpec:
    return ArenaEnvInitialGraphSpec.from_yaml(_VALID_SPEC_YAML_PATH)


@pytest.fixture
def session_state(monkeypatch):
    """Replace ``st.session_state`` with a plain dict for review-GUI unit tests."""
    state: dict = {}
    monkeypatch.setattr("streamlit.session_state", state, raising=False)
    return state


class TestValidateYamlText:
    @pytest.mark.parametrize("text", ["", "   \n  "], ids=["empty", "whitespace"])
    def test_blank_text_is_neutral(self, session_state, text: str):
        result = validate_yaml_text(text)
        assert result.spec is None
        assert result.error is None
        assert not result.is_valid
        assert session_state["_validation_text"] == text
        assert session_state["_validation_result"] is result

    def test_valid_spec_yaml(self, session_state, valid_spec_yaml: str, valid_spec: ArenaEnvInitialGraphSpec):
        result = validate_yaml_text(valid_spec_yaml)
        assert result.is_valid
        assert result.error is None
        assert result.spec is not None
        assert result.spec.env_name == valid_spec.env_name

    @pytest.mark.parametrize(
        ("text", "error_predicate"),
        [
            ("null\n", lambda error: error == "YAML is empty"),
            ("- not: a mapping\n", lambda error: "Expected mapping" in error),
            ("{unclosed", lambda error: error is not None and "Traceback" in error),
            ("env_name: broken\n", lambda error: error is not None),
        ],
        ids=["null_document", "non_mapping_root", "invalid_syntax", "invalid_schema"],
    )
    def test_rejects_invalid_yaml(self, session_state, text: str, error_predicate):
        result = validate_yaml_text(text)
        assert result.spec is None
        assert error_predicate(result.error)

    def test_caches_result_for_same_text(self, session_state, valid_spec_yaml: str):
        first = validate_yaml_text(valid_spec_yaml)
        with patch(
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel.ArenaEnvInitialGraphSpec.from_dict",
        ) as mock_from_dict:
            second = validate_yaml_text(valid_spec_yaml)
            mock_from_dict.assert_not_called()
        assert second is first


class TestParseArgs:
    def test_defaults_to_none_spec_path(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["streamlit_ui.py"])
        args = parse_args()
        assert args.env_initial_graph_spec is None

    def test_parses_env_initial_graph_spec(self, monkeypatch, tmp_path: Path):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("env_name: x\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["streamlit_ui.py", "--env_initial_graph_spec", str(spec_path)])
        args = parse_args()
        assert args.env_initial_graph_spec == spec_path


class TestInitializeState:
    def test_seeds_empty_session_without_yaml_path(self, session_state):
        initialize_state(None)
        assert session_state["_yaml_path"] == ""
        assert session_state["edited_text"] == ""
        assert session_state["save_path"] == DEFAULT_SAVE_PATH
        assert session_state["generation_prompt"] == DEFAULT_GENERATION_PROMPT
        assert session_state["editor_version"] == 0

    def test_loads_yaml_from_disk(self, session_state, valid_spec_yaml: str, tmp_path: Path):
        spec_path = tmp_path / "opened.yaml"
        spec_path.write_text(valid_spec_yaml, encoding="utf-8")
        initialize_state(spec_path)
        assert session_state["edited_text"] == valid_spec_yaml
        assert session_state["original_text"] == valid_spec_yaml
        assert session_state["save_path"] == str(spec_path)
        assert session_state["last_rendered_text"] == ""
        assert session_state["rendered_html"] == ""

    def test_skips_reinitialization_for_same_path(self, session_state, tmp_path: Path):
        spec_path = tmp_path / "opened.yaml"
        spec_path.write_text("env_name: first\n", encoding="utf-8")
        initialize_state(spec_path)
        session_state["edited_text"] = "user edits"
        spec_path.write_text("env_name: second\n", encoding="utf-8")
        initialize_state(spec_path)
        assert session_state["edited_text"] == "user edits"

    def test_reinitializes_when_path_changes(self, session_state, tmp_path: Path):
        first = tmp_path / "first.yaml"
        second = tmp_path / "second.yaml"
        first.write_text("env_name: first\n", encoding="utf-8")
        second.write_text("env_name: second\n", encoding="utf-8")
        initialize_state(first)
        initialize_state(second)
        assert session_state["_yaml_path"] == str(second.resolve())
        assert session_state["edited_text"] == "env_name: second\n"


class TestFormatTraceLines:
    def test_formats_all_events(self):
        trace = [
            {"stage": "asset.match", "query": "bowl", "chosen": "bowl_ycb", "note": ""},
            {"stage": "asset.no_match", "query": "missing", "chosen": None, "note": "fallback"},
        ]
        lines = _format_trace_lines(trace)
        assert "asset.match" in lines
        assert "bowl_ycb" in lines
        assert "<none>" in lines
        assert "[fallback]" in lines

    def test_errors_only_filters_non_error_stages(self):
        error_stage = next(iter(ASSET_ERROR_STAGES))
        trace = [
            {"stage": "asset.match", "query": "bowl", "chosen": "bowl_ycb", "note": ""},
            {"stage": error_stage, "query": "missing", "chosen": None, "note": ""},
        ]
        lines = _format_trace_lines(trace, errors_only=True)
        assert error_stage in lines
        assert "asset.match" not in lines


class TestApplyGeneratedYaml:
    def test_with_spec_updates_editor_and_validation_cache(self, session_state, valid_spec: ArenaEnvInitialGraphSpec):
        session_state["editor_version"] = 2
        yaml_text = yaml.safe_dump(valid_spec.to_dict(), sort_keys=False)
        with patch(
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.render_dashboard_html",
            return_value="<html>preview</html>",
        ) as mock_render:
            _apply_generated_yaml(yaml_text, spec=valid_spec)
        mock_render.assert_called_once_with(valid_spec)
        assert session_state["edited_text"] == yaml_text
        assert session_state["editor_version"] == 3
        assert session_state["last_rendered_text"] == yaml_text
        assert session_state["rendered_html"] == "<html>preview</html>"
        assert session_state["_validation_text"] == yaml_text
        assert session_state["_validation_result"].spec is valid_spec

    def test_without_spec_clears_preview_and_validation_cache(self, session_state):
        session_state["_validation_text"] = "old"
        session_state["_validation_result"] = SpecParseResult(spec=None, error="old")
        session_state["rendered_html"] = "<html>old</html>"
        _apply_generated_yaml("edited:\n  yaml: true\n", spec=None)
        assert session_state["edited_text"] == "edited:\n  yaml: true\n"
        assert session_state["rendered_html"] == ""
        assert "_validation_text" not in session_state
        assert "_validation_result" not in session_state


class TestRunGenerationPipeline:
    def test_rejects_empty_prompt(self, session_state):
        ok, message = run_generation_pipeline("   ")
        assert not ok
        assert "Enter a prompt" in message

    def test_fails_when_agent_unavailable(self, session_state):
        session_state["generation_agent_error"] = "missing key"
        ok, message = run_generation_pipeline("pick up a cube")
        assert not ok
        assert "missing key" in message

    def test_fails_when_catalogue_build_raises(self, session_state):
        session_state["generation_agent"] = MagicMock()
        with patch(
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
            side_effect=RuntimeError("registry unavailable"),
        ):
            ok, message = run_generation_pipeline("pick up a cube")
        assert not ok
        assert "registry unavailable" in message

    def test_success_loads_generated_yaml_into_session(self, session_state, valid_spec: ArenaEnvInitialGraphSpec):
        mock_agent = MagicMock()
        mock_intent = MagicMock(reasoning="picked assets")
        mock_agent.generate_spec.return_value = (mock_intent, "{}")
        session_state["generation_agent"] = mock_agent

        mock_catalogues = MagicMock()
        mock_compiler = MagicMock()
        mock_compiler.compile.return_value = valid_spec
        mock_compiler.trace = []
        mock_compiler.has_resolution_errors = False

        with (
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
                return_value=mock_catalogues,
            ),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.IntentCompiler",
                return_value=mock_compiler,
            ),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.render_dashboard_html",
                return_value="<html>generated</html>",
            ),
        ):
            ok, message = run_generation_pipeline("pick up a cube")

        assert ok
        assert "loaded into the YAML editor" in message
        assert session_state["last_generation_reasoning"] == "picked assets"
