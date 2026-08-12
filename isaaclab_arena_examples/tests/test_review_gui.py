# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib
import sys
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    INFERENCE_ENDPOINT_ENV_VAR,
    INFERENCE_ENDPOINTS,
    INTERNAL_ENDPOINT,
    OPENAI_ENDPOINT,
    PUBLIC_ENDPOINT,
)
from isaaclab_arena.agentic_environment_generation.simready_asset_search import SimReadySearchConfig
from isaaclab_arena.agentic_environment_generation.spec_io import env_graph_spec_path, write_env_graph_spec
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord
from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import (
    SpecParseResult,
    try_save_env_graph_spec,
    validate_yaml_text,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel import (
    DEFAULT_GENERATION_PROMPT,
    _apply_generated_yaml,
    _clear_orphaned_generation_agents,
    _default_inference_endpoint,
    _generation_agent_cache_key,
    _get_generation_agent,
    available_inference_endpoints,
    run_generation_pipeline,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppClient,
    spawn_simapp_process,
    stop_simapp_process,
    wait_for_simapp_socket,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.sim_preview import (
    _preview_cfg,
    parse_sim_preview_params,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import (
    ENV_SPACING_M,
    NUM_ENVS,
    NUM_STEPS,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.asset_cards import (
    build_asset_cards,
    object_set_member_key,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.mermaid_graph import (
    estimate_mermaid_height_px,
    render_mermaid_graph,
    render_mermaid_html,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.prim_tree_view import (
    build_prim_nodes,
    render_prim_tree_html,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.streamlit_ui import initialize_state, parse_args
from isaaclab_arena_examples.agentic_environment_generation.review_gui.visualization_service import (
    resolve_background_prim_tree,
)

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
    def test_preview_cfg_uses_cpu_without_fabric(self):
        cfg = _preview_cfg(num_envs=1, env_spacing=30.0)
        assert cfg.device == "cpu"
        assert cfg.disable_fabric

    def test_parse_sim_preview_params_requires_all_keys(self):
        with pytest.raises(ValueError, match="missing required sim preview params"):
            parse_sim_preview_params({})

    def test_parse_sim_preview_params_custom(self):
        assert parse_sim_preview_params({"num_envs": 8, "num_steps": 3, "env_spacing": 2.0}) == (8, 3, 2.0)

    def test_parse_sim_preview_params_rejects_invalid(self):
        with pytest.raises(AssertionError):
            parse_sim_preview_params({"num_envs": 0, "num_steps": 10, "env_spacing": 1.5})
        with pytest.raises(AssertionError):
            parse_sim_preview_params({"num_envs": 1, "num_steps": 0, "env_spacing": 1.5})


class TestBuildAssetCards:
    def test_attaches_snapshot_and_aabb(self, valid_spec: ArenaEnvGraphSpec):
        bg_id = valid_spec.background.id
        cards = build_asset_cards(
            valid_spec,
            thumbnails={bg_id: b"fake"},
            aabb_dimensions_m={bg_id: (0.05, 0.05, 0.12)},
        )
        background = next(card for card in cards if card.spec.id == bg_id)
        assert background.thumbnail_bytes == b"fake"
        assert background.aabb_dimensions_m == (0.05, 0.05, 0.12)

    def test_includes_object_references(self):
        from isaaclab_arena.tests.utils.agentic_environment_generation import kitchen_pass1_dict

        spec = ArenaEnvGraphSpec.model_validate(kitchen_pass1_dict())
        card_ids = {card.spec.id for card in build_asset_cards(spec)}
        assert card_ids
        assert any(ref.id in card_ids for ref in spec.object_references)

    def test_object_set_yields_one_card_per_member(self):
        spec = ArenaEnvGraphSpec.from_yaml(
            _REPO_ROOT / "isaaclab_arena/tests/test_data/object_set_maple_table_env_graph.yaml"
        )
        (object_set,) = spec.object_sets
        sweet_potato_key = object_set_member_key(object_set.id, "sweet_potato")
        cards = build_asset_cards(
            spec,
            thumbnails={sweet_potato_key: b"fake"},
            aabb_dimensions_m={sweet_potato_key: (0.1, 0.1, 0.2)},
        )

        member_cards = [card for card in cards if card.role == "object_set"]
        assert [card.spec.registry_name for card in member_cards] == object_set.members
        assert all(card.spec.id == object_set.id for card in member_cards)

        # Snapshots are keyed per member, so one member's thumbnail never leaks onto its siblings.
        sweet_potato = next(card for card in member_cards if card.spec.registry_name == "sweet_potato")
        assert sweet_potato.thumbnail_bytes == b"fake"
        assert sweet_potato.aabb_dimensions_m == (0.1, 0.1, 0.2)
        jug = next(card for card in member_cards if card.spec.registry_name == "jug")
        assert jug.thumbnail_bytes is None


class TestMermaidHtml:
    def test_render_mermaid_html_includes_syntax_and_initialize(self, valid_spec: ArenaEnvGraphSpec):
        html = render_mermaid_html(valid_spec)
        assert "mermaid.initialize" in html
        assert html_lib.escape(render_mermaid_graph(valid_spec)) in html

    def test_estimate_mermaid_height_px_scales_with_nodes(self, valid_spec: ArenaEnvGraphSpec):
        height = estimate_mermaid_height_px(valid_spec)
        assert 260 <= height <= 900


class TestPrimTreeView:
    _TREE = [
        UsdPrimRecord("cab_1_main_group", ObjectType.ARTICULATION, ("right_door_joint",)),
        UsdPrimRecord("cab_1_main_group/corpus", ObjectType.RIGID),
        UsdPrimRecord("cab_1_main_group/corpus/back", ObjectType.BASE),
    ]

    def test_build_prim_nodes_nests_by_relative_path(self):
        roots = build_prim_nodes(self._TREE)
        assert len(roots) == 1
        assert roots[0].text == "cab_1_main_group (articulation right_door_joint)"
        assert roots[0].children[0].text == "corpus (rigid)"
        assert roots[0].children[0].children[0].text == "back (base)"

    def test_render_prim_tree_html_includes_search_and_nodes(self):
        markup = render_prim_tree_html(self._TREE)
        assert 'id="search"' in markup
        assert "corpus (rigid)" in markup
        assert "cab_1_main_group/corpus" in markup


class TestBackgroundPrimTree:
    def test_returns_loaded_prim_tree_records(self, monkeypatch):
        from isaaclab_arena.tests.utils.agentic_environment_generation import kitchen_pass1_dict, kitchen_prim_tree

        spec = ArenaEnvGraphSpec.model_validate(kitchen_pass1_dict())
        monkeypatch.setattr(
            "isaaclab_arena.environment_spec.arena_env_graph_types.AssetSpec.resolve_usd_path",
            lambda self, *_args, **_kwargs: "/tmp/scene.usd",
        )
        monkeypatch.setattr(
            "isaaclab_arena.utils.usd_prim_tree.load_usd_prim_tree",
            lambda *_args, **_kwargs: kitchen_prim_tree(),
        )
        prim_tree = resolve_background_prim_tree(spec)
        assert any(record.relative_path == "fridge_main_group" for record in prim_tree)


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
        assert args.env_spec is None

    def test_parses_env_spec(self, monkeypatch, tmp_path: Path):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("env_name: x\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["streamlit_ui.py", "--env_spec", str(spec_path)])
        args = parse_args()
        assert args.env_spec == spec_path

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
        assert session_state["rendered_visualization"] is None

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
        assert session_state["rendered_visualization"] is None
        assert session_state["_defer_viz_render"] is True
        assert session_state["_validation_text"] == yaml_text
        assert session_state["_validation_result"].spec is valid_spec

    def test_without_spec_clears_preview_and_validation_cache(self, session_state):
        session_state["_validation_text"] = "old"
        session_state["_validation_result"] = SpecParseResult(spec=None, error="old")
        session_state["rendered_visualization"] = ["stale"]
        _apply_generated_yaml("edited:\n  yaml: true\n", spec=None)
        assert session_state["edited_text"] == "edited:\n  yaml: true\n"
        assert session_state["rendered_visualization"] is None
        assert "_validation_text" not in session_state
        assert "_validation_result" not in session_state


def _patch_generation_agent(agent: MagicMock | None):
    """Stub the panel's agent accessor.

    The real one caches the agent under a key built from the SimReady settings, so seeding a
    session-state entry by name does not reach it.
    """
    return patch(
        "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel._get_generation_agent",
        return_value=agent,
    )


class TestInferenceEndpointSelection:
    @pytest.fixture(autouse=True)
    def clean_endpoint_env(self, monkeypatch):
        monkeypatch.delenv(INFERENCE_ENDPOINT_ENV_VAR, raising=False)
        for endpoint in INFERENCE_ENDPOINTS.values():
            monkeypatch.delenv(endpoint.api_key_env_var, raising=False)

    def test_available_endpoints_omit_unset_keys(self, monkeypatch):
        monkeypatch.setenv(INTERNAL_ENDPOINT.api_key_env_var, "internal-key")
        assert available_inference_endpoints() == [INTERNAL_ENDPOINT.name]

    def test_available_endpoints_include_every_set_key(self, monkeypatch):
        monkeypatch.setenv(INTERNAL_ENDPOINT.api_key_env_var, "internal-key")
        monkeypatch.setenv(PUBLIC_ENDPOINT.api_key_env_var, "public-key")
        monkeypatch.setenv(OPENAI_ENDPOINT.api_key_env_var, "openai-key")
        assert available_inference_endpoints() == [
            INTERNAL_ENDPOINT.name,
            PUBLIC_ENDPOINT.name,
            OPENAI_ENDPOINT.name,
        ]

    def test_default_falls_back_when_preferred_key_missing(self, monkeypatch):
        monkeypatch.setenv(INFERENCE_ENDPOINT_ENV_VAR, PUBLIC_ENDPOINT.name)
        monkeypatch.setenv(INTERNAL_ENDPOINT.api_key_env_var, "internal-key")
        available = available_inference_endpoints()
        assert PUBLIC_ENDPOINT.name not in available
        assert _default_inference_endpoint(available) == INTERNAL_ENDPOINT.name

    def test_default_prefers_arena_inference_endpoint_when_available(self, monkeypatch):
        monkeypatch.setenv(INFERENCE_ENDPOINT_ENV_VAR, OPENAI_ENDPOINT.name)
        monkeypatch.setenv(INTERNAL_ENDPOINT.api_key_env_var, "internal-key")
        monkeypatch.setenv(OPENAI_ENDPOINT.api_key_env_var, "openai-key")
        available = available_inference_endpoints()
        assert _default_inference_endpoint(available) == OPENAI_ENDPOINT.name

    def test_cache_key_includes_endpoint(self):
        cfg = SimReadySearchConfig()
        public_key = _generation_agent_cache_key(PUBLIC_ENDPOINT.name, simready_enabled=False, simready_config=cfg)
        internal_key = _generation_agent_cache_key(INTERNAL_ENDPOINT.name, simready_enabled=False, simready_config=cfg)
        assert public_key != internal_key
        assert PUBLIC_ENDPOINT.name in public_key
        assert INTERNAL_ENDPOINT.name in internal_key

    def test_clear_orphaned_generation_agents_keeps_requested_key(self, session_state):
        cfg = SimReadySearchConfig()
        keep = _generation_agent_cache_key(PUBLIC_ENDPOINT.name, simready_enabled=False, simready_config=cfg)
        other = _generation_agent_cache_key(INTERNAL_ENDPOINT.name, simready_enabled=False, simready_config=cfg)
        session_state[keep] = object()
        session_state[other] = object()
        session_state["unrelated"] = "keep-me"
        _clear_orphaned_generation_agents(keep=keep)
        assert keep in session_state
        assert other not in session_state
        assert session_state["unrelated"] == "keep-me"

    def test_get_generation_agent_retries_after_failed_init(self, session_state, monkeypatch):
        monkeypatch.setenv(PUBLIC_ENDPOINT.api_key_env_var, "public-key")
        session_state["inference_endpoint"] = PUBLIC_ENDPOINT.name
        session_state["generation_agent_error"] = "previous failure"
        mock_agent = MagicMock(name="generation-agent")
        with patch(
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.EnvironmentGenerationAgent",
            side_effect=[AssertionError("transient"), mock_agent],
        ) as mock_cls:
            assert _get_generation_agent() is None
            assert session_state["generation_agent_error"] == "transient"
            assert _get_generation_agent() is mock_agent
        assert mock_cls.call_count == 2
        assert "generation_agent_error" not in session_state
        assert mock_cls.call_args.kwargs["endpoint"] == PUBLIC_ENDPOINT.name


class TestRunGenerationPipeline:
    def test_rejects_empty_prompt(self, session_state):
        ok, message = run_generation_pipeline("   ")
        assert not ok
        assert "Enter a prompt" in message

    def test_fails_when_agent_unavailable(self, session_state):
        session_state["generation_agent_error"] = "missing key"
        with _patch_generation_agent(None):
            ok, message = run_generation_pipeline("pick up a cube")
        assert not ok
        assert "missing key" in message

    def test_fails_when_catalogue_build_raises(self, session_state):
        with (
            _patch_generation_agent(MagicMock()),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
                side_effect=RuntimeError("registry unavailable"),
            ),
        ):
            ok, message = run_generation_pipeline("pick up a cube")
        assert not ok
        assert "registry unavailable" in message

    def test_success_loads_generated_yaml_into_session(
        self, session_state, valid_spec: ArenaEnvGraphSpec, tmp_path: Path
    ):
        session_state["out_dir"] = str(tmp_path)
        mock_agent = MagicMock()
        mock_agent.generate_spec.return_value = (valid_spec, None)
        # Explicit, because a MagicMock attribute is truthy and would read as a missing asset.
        mock_agent.unavailable_objects = ()

        with (
            _patch_generation_agent(mock_agent),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
                return_value=MagicMock(),
            ),
        ):
            ok, message = run_generation_pipeline("pick up a cube")

        assert ok
        assert "loaded into the YAML editor" in message
        assert session_state["_generation_severity"] == "success"
        assert session_state["save_path"]
        assert Path(session_state["save_path"]).is_file()

    def test_names_the_objects_no_asset_was_found_for(
        self, session_state, valid_spec: ArenaEnvGraphSpec, tmp_path: Path
    ):
        session_state["out_dir"] = str(tmp_path)
        mock_agent = MagicMock()
        # The spec is valid: the object nothing was found for was never offered to spec inference.
        mock_agent.generate_spec.return_value = (valid_spec, None)
        mock_agent.unavailable_objects = ("green trash can",)
        # Deliberately non-empty, though a successful generation leaves it empty: a success banner
        # must not dump traces at the user, because the GUI renders them as a warning.
        mock_agent.traces = ("a line the banner has no business showing",)

        with (
            _patch_generation_agent(mock_agent),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
                return_value=MagicMock(),
            ),
        ):
            ok, message = run_generation_pipeline("pick up a cube")

        assert ok
        # The banner names what the prompt asked for and did not get, so the swap is not silent.
        assert "No asset was found for: green trash can" in message
        assert "built without them" in message
        assert "a line the banner has no business showing" not in message
        # A silently substituted object is not a clean success, so the banner must not be green.
        assert session_state["_generation_severity"] == "warning"
        assert session_state["edited_text"]
        assert session_state["save_path"]

    def test_save_failure_still_reports_success(self, session_state, valid_spec: ArenaEnvGraphSpec, tmp_path: Path):
        session_state["out_dir"] = str(tmp_path)
        mock_agent = MagicMock()
        mock_agent.generate_spec.return_value = (valid_spec, None)
        # Explicit, because a MagicMock attribute is truthy and would read as a missing asset.
        mock_agent.unavailable_objects = ()

        with (
            _patch_generation_agent(mock_agent),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.get_catalogue_bundle",
                return_value=MagicMock(),
            ),
            patch(
                "isaaclab_arena_examples.agentic_environment_generation.review_gui.generation_panel.try_save_env_graph_spec",
                return_value=(None, "Save failed: disk full"),
            ),
        ):
            ok, message = run_generation_pipeline("pick up a cube")

        assert ok
        assert "save failed" in message.lower()
        assert session_state["_generation_severity"] == "warning"
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
                num_steps=2,
                env_spacing=ENV_SPACING_M,
            )
            assert response["ok"] is True

            video_path = Path(response["video_path"])
            assert video_path.is_file() and video_path.stat().st_size > 0
            assert response["num_envs"] == 1
            assert response["env_spacing"] == ENV_SPACING_M
            assert response["num_steps"] == 2
            assert client.ping()

            video_path.unlink()
            video_path.parent.rmdir()

            client.shutdown()
        finally:
            stop_simapp_process(proc, str(socket_path))
