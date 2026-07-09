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

from isaaclab_arena.agentic_environment_generation.spec_io import env_graph_spec_path, write_env_graph_spec
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (
    SpecParseResult,
    try_save_env_graph_spec,
    validate_yaml_text,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel import (
    DEFAULT_GENERATION_PROMPT,
    _apply_generated_yaml,
    run_generation_pipeline,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.render.thumbnails import (
    format_aabb_dimensions_m,
    render_asset_thumbnail,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppClient,
    spawn_simapp_process,
    stop_simapp_process,
    wait_for_simapp_socket,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.sim_preview import (
    _preview_args,
    parse_sim_preview_params,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import (
    ENV_SPACING_M,
    NUM_ENVS,
    NUM_STEPS,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.streamlit_ui import initialize_state, parse_args

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_SPEC_YAML_PATH = _REPO_ROOT / "isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml"


@pytest.fixture
def valid_spec_yaml() -> str:
    return _VALID_SPEC_YAML_PATH.read_text(encoding="utf-8")


@pytest.fixture
def valid_spec(valid_spec_yaml: str) -> ArenaEnvGraphSpec:
    return ArenaEnvGraphSpec.from_yaml(_VALID_SPEC_YAML_PATH)


@pytest.fixture
def session_state(monkeypatch):
    """Replace ``st.session_state`` with a plain dict for review-GUI unit tests."""
    state: dict = {}
    monkeypatch.setattr("streamlit.session_state", state, raising=False)
    return state


class TestSimPreviewParams:
    def test_preview_args_honor_gui_overrides(self):
        args = _preview_args(num_envs=4, env_spacing=2.5)
        assert args.num_envs == 4
        assert args.env_spacing == 2.5

    def test_parse_sim_preview_params_requires_all_keys(self):
        with pytest.raises(ValueError, match="missing required sim preview params"):
            parse_sim_preview_params({})

    def test_parse_sim_preview_params_custom(self):
        assert parse_sim_preview_params({"num_envs": 8, "num_steps": 3, "env_spacing": 2.0}) == (8, 3, 2.0)

    def test_parse_sim_preview_params_rejects_invalid(self):
        with pytest.raises(AssertionError):
            parse_sim_preview_params({"num_envs": 0, "num_steps": 10, "env_spacing": 1.5})


class TestNodeThumbnailAabb:
    def test_format_aabb_dimensions_m(self):
        assert format_aabb_dimensions_m((0.1, 0.2, 0.3)) == "0.100 × 0.200 × 0.300 m"

    def test_render_asset_thumbnail_includes_aabb_under_snapshot(self):
        html = render_asset_thumbnail("mug_ycb_robolab", png_bytes=b"fake", aabb_dimensions_m=(0.05, 0.05, 0.12))
        assert "thumb-dims" in html
        assert "0.050 × 0.050 × 0.120 m" in html
        assert html.index("thumb-wrap") < html.index("thumb-dims")

    def test_render_object_reference_shows_unsupported_note(self):
        html = render_asset_thumbnail("table_top", is_object_reference=True)
        assert "thumb-unsupported" in html
        assert "Prim reference — snapshot not supported" in html


class TestValidateYamlText:
    @pytest.mark.parametrize("text", ["", "   \n  "], ids=["empty", "whitespace"])
    def test_blank_text_is_neutral(self, session_state, text: str):
        result = validate_yaml_text(text)
        assert result.spec is None
        assert result.error is None
        assert not result.is_valid
        assert session_state["_validation_text"] == text
        assert session_state["_validation_result"] is result

    def test_valid_spec_yaml(self, session_state, valid_spec_yaml: str, valid_spec: ArenaEnvGraphSpec):
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
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel.ArenaEnvGraphSpec.from_dict",
        ) as mock_from_dict:
            second = validate_yaml_text(valid_spec_yaml)
            mock_from_dict.assert_not_called()
        assert second is first


class TestParseArgs:
    def test_defaults_to_none_spec_path(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["streamlit_ui.py"])
        args = parse_args()
        assert args.env_graph_spec_yaml is None

    def test_parses_env_graph_spec_yaml(self, monkeypatch, tmp_path: Path):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("env_name: x\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["streamlit_ui.py", "--env_graph_spec_yaml", str(spec_path)])
        args = parse_args()
        assert args.env_graph_spec_yaml == spec_path

    def test_parses_out_dir(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(sys, "argv", ["streamlit_ui.py", "--out_dir", str(tmp_path / "generated")])
        args = parse_args()
        assert args.out_dir == tmp_path / "generated"


class TestInitializeState:
    def test_seeds_empty_session_without_yaml_path(self, session_state, tmp_path: Path):
        out_dir = tmp_path / "agent_generated"
        initialize_state(None, out_dir)
        assert session_state["_yaml_path"] == ""
        assert session_state["edited_text"] == ""
        assert session_state["save_path"] == ""
        assert session_state["out_dir"] == str(out_dir.resolve())
        assert session_state["generation_prompt"] == DEFAULT_GENERATION_PROMPT
        assert session_state["editor_version"] == 0
        assert session_state["sim_preview_num_envs"] == NUM_ENVS
        assert session_state["sim_preview_num_steps"] == NUM_STEPS
        assert session_state["sim_preview_env_spacing"] == ENV_SPACING_M

    def test_loads_yaml_from_disk(self, session_state, valid_spec_yaml: str, tmp_path: Path):
        spec_path = tmp_path / "opened.yaml"
        out_dir = tmp_path / "out"
        spec_path.write_text(valid_spec_yaml, encoding="utf-8")
        initialize_state(spec_path, out_dir)
        assert session_state["edited_text"] == valid_spec_yaml
        assert session_state["original_text"] == valid_spec_yaml
        assert session_state["save_path"] == str(spec_path)
        assert session_state["last_rendered_text"] == ""
        assert session_state["rendered_html"] == ""

    def test_skips_reinitialization_for_same_path(self, session_state, tmp_path: Path):
        spec_path = tmp_path / "opened.yaml"
        out_dir = tmp_path / "out"
        spec_path.write_text("env_name: first\n", encoding="utf-8")
        initialize_state(spec_path, out_dir)
        session_state["edited_text"] = "user edits"
        spec_path.write_text("env_name: second\n", encoding="utf-8")
        initialize_state(spec_path, out_dir)
        assert session_state["edited_text"] == "user edits"

    def test_reinitializes_when_path_changes(self, session_state, tmp_path: Path):
        first = tmp_path / "first.yaml"
        second = tmp_path / "second.yaml"
        out_dir = tmp_path / "out"
        first.write_text("env_name: first\n", encoding="utf-8")
        second.write_text("env_name: second\n", encoding="utf-8")
        initialize_state(first, out_dir)
        initialize_state(second, out_dir)
        assert session_state["_yaml_path"] == str(second.resolve())
        assert session_state["edited_text"] == "env_name: second\n"


class TestApplyGeneratedYaml:
    def test_with_spec_updates_editor_and_validation_cache(self, session_state, valid_spec: ArenaEnvGraphSpec):
        session_state["editor_version"] = 2
        yaml_text = yaml.safe_dump(valid_spec.to_dict(), sort_keys=False)
        _apply_generated_yaml(yaml_text, spec=valid_spec)
        assert session_state["edited_text"] == yaml_text
        assert session_state["editor_version"] == 3
        assert session_state["last_rendered_text"] == ""
        assert session_state["rendered_html"] == ""
        assert session_state["_defer_viz_render"] is True
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

    def test_success_loads_generated_yaml_into_session(
        self, session_state, valid_spec: ArenaEnvGraphSpec, tmp_path: Path
    ):
        session_state["out_dir"] = str(tmp_path)
        mock_agent = MagicMock()
        mock_agent.generate_spec.return_value = (valid_spec, "{}")
        session_state["generation_agent"] = mock_agent

        mock_catalogues = MagicMock()

        with patch(
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
            return_value=mock_catalogues,
        ):
            ok, message = run_generation_pipeline("pick up a cube")

        assert ok
        assert "loaded into the YAML editor" in message
        assert session_state["save_path"]
        assert Path(session_state["save_path"]).is_file()

    def test_save_failure_still_reports_success(self, session_state, valid_spec: ArenaEnvGraphSpec, tmp_path: Path):
        session_state["out_dir"] = str(tmp_path)
        mock_agent = MagicMock()
        mock_agent.generate_spec.return_value = (valid_spec, "{}")
        session_state["generation_agent"] = mock_agent

        mock_catalogues = MagicMock()

        with (
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
                return_value=mock_catalogues,
            ),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.try_save_env_graph_spec",
                return_value=(None, "Save failed: disk full"),
            ),
        ):
            ok, message = run_generation_pipeline("pick up a cube")

        assert ok
        assert "save failed" in message.lower()
        assert session_state["edited_text"]
        assert "save_path" not in session_state


class TestSaveEnvGraphSpec:
    def test_writes_graph_spec_yaml(self, valid_spec: ArenaEnvGraphSpec, tmp_path: Path):
        path = write_env_graph_spec(valid_spec, tmp_path)
        assert path == env_graph_spec_path(valid_spec.env_name, tmp_path)
        assert path.is_file()


class TestTrySaveEnvGraphSpec:
    def test_returns_error_when_save_fails(self, valid_spec: ArenaEnvGraphSpec, tmp_path: Path):
        with patch(
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel.write_env_graph_spec",
            side_effect=ValueError("unknown node reference"),
        ):
            path, error = try_save_env_graph_spec(valid_spec, tmp_path)
        assert path is None
        assert "ValueError" in error
        assert "unknown node reference" in error


class TestSimAppClient:
    def test_disconnect_leaves_server_listening(self, tmp_path: Path) -> None:
        """Boot probe must not send shutdown — Streamlit connects after wait_for_simapp_socket."""
        import json
        import socket
        import threading

        socket_path = tmp_path / "probe.sock"
        shutdowns = 0
        pings = 0

        def _serve() -> None:
            nonlocal shutdowns, pings
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(socket_path))
            server.listen(5)
            try:
                while True:
                    conn, _ = server.accept()
                    with conn:
                        reader = conn.makefile("r", encoding="utf-8", newline="\n")
                        writer = conn.makefile("w", encoding="utf-8", newline="\n")
                        for raw_line in reader:
                            req = json.loads(raw_line)
                            if req.get("cmd") == "shutdown":
                                shutdowns += 1
                                writer.write(json.dumps({"ok": True}) + "\n")
                                writer.flush()
                                return
                            if req.get("cmd") == "ping":
                                pings += 1
                                writer.write(json.dumps({"ok": True}) + "\n")
                                writer.flush()
            finally:
                server.close()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()

        class _Proc:
            def poll(self) -> None:
                return None

        wait_for_simapp_socket(str(socket_path), _Proc(), timeout_s=5.0, poll_interval_s=0.05)
        assert pings == 1
        assert shutdowns == 0

        client = SimAppClient.connect(str(socket_path))
        assert client.ping()
        client.disconnect()
        assert shutdowns == 0

        client = SimAppClient.connect(str(socket_path))
        client.shutdown()
        thread.join(timeout=2.0)
        assert shutdowns == 1


class TestSimAppSimPreview:
    @pytest.mark.with_subprocess
    def test_run_sim_preview_via_simapp_subprocess(self, tmp_path: Path) -> None:
        yaml_text = _VALID_SPEC_YAML_PATH.read_text(encoding="utf-8")
        socket_path = tmp_path / "sim_preview.sock"
        proc = spawn_simapp_process(str(socket_path))
        try:
            wait_for_simapp_socket(str(socket_path), proc, timeout_s=180.0, poll_interval_s=0.5)
            client = SimAppClient.connect(str(socket_path))
            response = client.run_sim_preview(
                yaml_text,
                num_envs=1,
                num_steps=0,
                env_spacing=1.5,
            )
            assert response["ok"] is True

            first_frame = Path(response["first_frame"])
            last_frame = Path(response["last_frame"])
            assert first_frame.is_file() and first_frame.stat().st_size > 0
            assert last_frame.is_file() and last_frame.stat().st_size > 0
            assert response["num_envs"] == 1
            assert response["num_steps"] == 0

            client.shutdown()
        finally:
            stop_simapp_process(proc, str(socket_path))
