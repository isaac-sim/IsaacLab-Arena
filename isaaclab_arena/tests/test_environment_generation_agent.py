# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    AssetCatalogue,
    EnvironmentGenerationAgent,
    RelationCatalogue,
    TaskCatalogue,
)
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import TaskCompositionType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _chat_response(content: str | None = None, reasoning_content: str | None = None, finish_reason: str = "stop"):
    """Build a nested mock matching the openai chat-completion response shape.

    Models that route structured outputs into ``reasoning_content`` (e.g.
    NVIDIA DeepSeek) leave ``content`` empty — the fixture mirrors that by
    populating either channel independently.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].finish_reason = finish_reason
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning_content
    return resp


@pytest.fixture
def stub_openai():
    """Patch ``openai.OpenAI`` so ``EnvironmentGenerationAgent()`` never hits the wire."""
    with patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.OpenAI") as mock_cls:
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content="OK")
        mock_cls.return_value = client
        yield mock_cls


@pytest.fixture
def agent(stub_openai):
    """A constructed ``EnvironmentGenerationAgent`` with a fully mocked openai client."""
    a = EnvironmentGenerationAgent(api_key="test-key")
    a.client.chat.completions.create.side_effect = None
    a.client.chat.completions.create.reset_mock()
    return a


# Minimal ArenaEnvGraphSpec payload — uses registered asset names from the test fixture.
_MINIMAL_SPEC: dict = {
    "env_name": "llm_gen_maple_table_robolab_PickAndPlaceTask",
    "embodiment": {"id": "franka_ik", "registry_name": "franka_ik"},
    "background": {"id": "maple_table_robolab", "registry_name": "maple_table_robolab"},
    "objects": [
        {"id": "rubiks_cube_hot3d_robolab", "registry_name": "rubiks_cube_hot3d_robolab"},
        {"id": "bowl_ycb_robolab", "registry_name": "bowl_ycb_robolab"},
    ],
    "object_references": [
        {
            "id": "maple_table_robolab_table",
            "parent_id": "maple_table_robolab",
            "prim_path": "{ENV_REGEX_NS}/maple_table_robolab/table",
            "object_type": "rigid",
        },
    ],
    "relations": [
        {"kind": "is_anchor", "subject": "maple_table_robolab_table"},
        {"kind": "on", "subject": "rubiks_cube_hot3d_robolab", "reference": "maple_table_robolab_table"},
        {"kind": "on", "subject": "bowl_ycb_robolab", "reference": "maple_table_robolab_table"},
    ],
    "task": {
        "composition": "atomic",
        "description": "pick up the rubiks cube and place it in the bowl",
        "subtasks": [{
            "kind": "PickAndPlaceTask",
            "params": {
                "pick_up_object": "rubiks_cube_hot3d_robolab",
                "destination_location": "bowl_ycb_robolab",
                "background_scene": "maple_table_robolab",
            },
        }],
    },
}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_explicit_api_key_overrides_env(self, monkeypatch, stub_openai):
        monkeypatch.setenv("NV_API_KEY", "env-key")
        a = EnvironmentGenerationAgent(api_key="explicit-key")
        assert a.api_key == "explicit-key"

    def test_falls_back_to_env_var(self, monkeypatch, stub_openai):
        monkeypatch.setenv("NV_API_KEY", "env-key")
        a = EnvironmentGenerationAgent()
        assert a.api_key == "env-key"

    def test_raises_when_no_key_anywhere(self, monkeypatch, stub_openai):
        monkeypatch.delenv("NV_API_KEY", raising=False)
        with pytest.raises(AssertionError, match="API key required"):
            EnvironmentGenerationAgent()

    def test_custom_model_and_base_url(self, stub_openai):
        a = EnvironmentGenerationAgent(api_key="k", model="custom-model", base_url="http://localhost:8000")
        assert a.model == "custom-model"
        stub_openai.assert_called_once_with(api_key="k", base_url="http://localhost:8000")


# ---------------------------------------------------------------------------
# generate_spec
# ---------------------------------------------------------------------------


def _catalog(text: str, relation_text: str = "RELATIONS (1):\n- on (binary): test") -> AssetCatalogue:
    catalogue = AssetCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def _relation_catalog(text: str) -> RelationCatalogue:
    catalogue = RelationCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def _task_catalog(text: str) -> TaskCatalogue:
    catalogue = TaskCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


class TestGenerateSpec:
    def test_builds_catalogues_from_singleton_registries_when_none(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        with (
            patch(
                "isaaclab_arena.agentic_environment_generation.environment_generation_agent.build_asset_catalogue",
            ) as mock_build_assets,
            patch(
                "isaaclab_arena.agentic_environment_generation.environment_generation_agent.build_relation_catalogue",
            ) as mock_build_relations,
            patch(
                "isaaclab_arena.agentic_environment_generation.environment_generation_agent.build_task_catalogue",
            ) as mock_build_tasks,
        ):
            mock_build_assets.return_value = _catalog("<<ASSET-CATALOG>>")
            mock_build_relations.return_value = _relation_catalog("<<RELATION-CATALOG>>")
            mock_build_tasks.return_value = _task_catalog("<<TASK-CATALOG>>")
            agent.generate_spec("p")
        mock_build_assets.assert_called_once_with()
        mock_build_relations.assert_called_once_with()
        mock_build_tasks.assert_called_once_with()

    def test_request_sets_response_format_to_json_schema(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec(
            "p",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
        )
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "ArenaEnvGraphSpec"
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        assert kwargs["response_format"]["json_schema"]["schema"] is agent._spec_schema

    def test_tolerates_unescaped_control_chars(self, agent):
        payload = dict(_MINIMAL_SPEC)
        payload["env_name"] = "pick\tup"
        raw = json.dumps(payload).replace("\\t", "\t")
        assert "\t" in raw
        agent.client.chat.completions.create.return_value = _chat_response(content=raw)
        spec, _ = agent.generate_spec(
            "p",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
        )
        assert spec is not None
        assert "\t" in spec.env_name

    def test_user_message_contains_catalog_and_prompt(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec(
            "user wants avocado on kitchen",
            asset_catalog=_catalog("<<CATALOG-MARKER>>"),
            relation_catalog=_relation_catalog("<<RELATIONS-MARKER>>"),
            task_catalog=_task_catalog("<<TASKS-MARKER>>"),
        )
        msgs = agent.client.chat.completions.create.call_args.kwargs["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        user_msg = msgs[1]["content"]
        assert "<<CATALOG-MARKER>>" in user_msg
        assert "<<RELATIONS-MARKER>>" in user_msg
        assert "<<TASKS-MARKER>>" in user_msg
        assert "user wants avocado on kitchen" in user_msg

    def test_raises_when_response_has_no_choices(self, agent):
        resp = MagicMock()
        resp.choices = []
        agent.client.chat.completions.create.return_value = resp
        with pytest.raises(RuntimeError, match="failed after 4 attempts"):
            agent.generate_spec(
                "p",
                asset_catalog=_catalog("catalog"),
                relation_catalog=_relation_catalog("RELATIONS"),
                task_catalog=_task_catalog("TASKS"),
                max_retries=3,
            )
        assert agent.client.chat.completions.create.call_count == 4

    def test_retries_after_api_error_then_succeeds(self, agent):
        agent.client.chat.completions.create.side_effect = [
            ConnectionError("timeout"),
            _chat_response(content=json.dumps(_MINIMAL_SPEC)),
        ]
        spec, _ = agent.generate_spec(
            "p",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
            max_retries=3,
        )
        assert spec is not None
        assert spec.background.registry_name == "maple_table_robolab"
        assert agent.client.chat.completions.create.call_count == 2

    def test_raises_after_api_errors_exhaust_retries(self, agent):
        agent.client.chat.completions.create.side_effect = ConnectionError("timeout")
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            agent.generate_spec(
                "p",
                asset_catalog=_catalog("catalog"),
                relation_catalog=_relation_catalog("RELATIONS"),
                task_catalog=_task_catalog("TASKS"),
                max_retries=1,
            )
        assert agent.client.chat.completions.create.call_count == 2

    def test_returns_none_with_validation_traces_on_invalid_spec(self, agent):
        invalid = dict(_MINIMAL_SPEC)
        invalid["embodiment"]["registry_name"] = "not_a_real_asset"
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(invalid))
        spec, data = agent.generate_spec(
            "p",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
        )
        assert isinstance(data, dict)
        assert data["embodiment"]["registry_name"] == "not_a_real_asset"
        assert spec is None
        assert agent.last_validation_traces
        assert any("registry_name" in line for line in agent.last_validation_traces)


# ---------------------------------------------------------------------------
# Live endpoint (network + auth required)
# ---------------------------------------------------------------------------

_ATOMIC_PICK_AND_PLACE_PROMPT = "Franka picks up the avocado and place it in the bowl on the maple table"

_FIVE_BANANAS_PROMPT = (
    "There are five bananas and a grey bin on the maple table. Droid places all the bananas into the bin."
)


def _assert_atomic_pick_and_place_spec(spec: ArenaEnvGraphSpec) -> None:
    """Check a single-object pick-and-place atomic task layout."""
    assert len(spec.objects) == 2, f"expected 2 objects, got {len(spec.objects)}"

    is_anchor = [relation for relation in spec.relations if relation.kind == "is_anchor"]
    assert len(is_anchor) == 1, f"expected one is_anchor relation, got {len(is_anchor)}"
    assert is_anchor[0].subject == spec.background.id

    object_ids = {obj.id for obj in spec.objects}
    on_subjects = {relation.subject for relation in spec.relations if relation.kind == "on"}
    assert on_subjects == object_ids

    assert spec.task.composition is TaskCompositionType.ATOMIC
    assert len(spec.task.subtasks) == 1
    assert spec.task.subtasks[0].kind == "PickAndPlaceTask"


def _assert_five_bananas_parallel_pick_and_place_spec(spec: ArenaEnvGraphSpec) -> None:
    """Check the five-bananas-into-bin parallel composite task layout."""
    assert len(spec.objects) == 6, f"expected 6 objects, got {len(spec.objects)}"

    object_ids = {obj.id for obj in spec.objects}
    on_subjects = {relation.subject for relation in spec.relations if relation.kind == "on"}
    for obj_id in object_ids:
        assert obj_id in on_subjects, f"object {obj_id!r} missing 'on' relation"

    assert spec.task.composition is TaskCompositionType.PARALLEL
    assert len(spec.task.subtasks) == 5

    pick_ids: list[str] = []
    dest_ids: list[str] = []
    for leaf in spec.task.subtasks:
        assert leaf.kind == "PickAndPlaceTask"
        pick_ids.append(leaf.params["pick_up_object"])
        dest_ids.append(leaf.params["destination_location"])

    assert len(set(pick_ids)) == 5, f"expected 5 distinct pick objects, got {pick_ids!r}"
    assert len(set(dest_ids)) == 1, f"expected one shared destination, got {dest_ids!r}"
    bin_id = dest_ids[0]
    assert bin_id not in pick_ids, f"destination {bin_id!r} should not be among pick objects"


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_atomic_pick_and_place_against_live_endpoint():
    """Live test: avocado into bowl yields an atomic pick-and-place task."""
    agent = EnvironmentGenerationAgent()
    spec, data = agent.generate_spec(_ATOMIC_PICK_AND_PLACE_PROMPT)
    assert spec is not None, f"spec validation failed: {agent.last_validation_traces}"
    assert isinstance(data, dict) and data, "agent returned empty parsed response"
    _assert_atomic_pick_and_place_spec(spec)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_five_bananas_parallel_pick_and_place_against_live_endpoint():
    """Live test: five bananas into one bin yields a parallel composite task."""
    agent = EnvironmentGenerationAgent()
    spec, data = agent.generate_spec(_FIVE_BANANAS_PROMPT)
    assert spec is not None, f"spec validation failed: {agent.last_validation_traces}"
    assert isinstance(data, dict) and data, "agent returned empty parsed response"
    _assert_five_bananas_parallel_pick_and_place_spec(spec)
