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
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import TaskCompositionType
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord

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
    "relations": [
        {"kind": "is_anchor", "subject": "maple_table_robolab"},
        {"kind": "on", "subject": "rubiks_cube_hot3d_robolab", "reference": "maple_table_robolab"},
        {"kind": "on", "subject": "bowl_ycb_robolab", "reference": "maple_table_robolab"},
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

_KITCHEN_PASS1: dict = {
    "env_name": "llm_gen_kitchen_pick_and_open",
    "embodiment": {"id": "droid_abs_joint_pos", "registry_name": "droid_abs_joint_pos"},
    "background": {
        "id": "lightwheel_robocasa_kitchen",
        "registry_name": "lightwheel_robocasa_kitchen",
        "params": {"layout_id": 1, "style_id": 1},
    },
    "objects": [
        {"id": "avocado", "registry_name": "avocado01_fruits_veggies_robolab"},
        {"id": "plate", "registry_name": "plate_large_vomp_robolab"},
    ],
    "object_references": [
        {
            "id": "counter_top",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "unknown",
            "object_type": "base",
        },
        {
            "id": "fridge",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "unknown",
            "object_type": "articulation",
        },
    ],
    "relations": [
        {"kind": "is_anchor", "subject": "counter_top"},
        {"kind": "on", "subject": "avocado", "reference": "counter_top"},
        {"kind": "on", "subject": "plate", "reference": "counter_top"},
    ],
    "tasks": [
        {
            "kind": "PickAndPlaceTask",
            "params": {
                "pick_up_object": "avocado",
                "destination_location": "plate",
                "background_scene": "lightwheel_robocasa_kitchen",
            },
            "description": "pick avocado and place on plate",
        },
        {
            "kind": "OpenDoorTask",
            "params": {"openable_object": "fridge"},
            "description": "open the fridge door",
        },
    ],
}

_KITCHEN_PRIM_TREE = [
    UsdPrimRecord("counter_right_main_group/top_geometry", "base"),
    UsdPrimRecord("fridge_main_group", "articulation", ("fridge_door_joint",)),
]

_RESOLVE_RESPONSE: dict = {
    "object_references": [
        {
            "id": "counter_top",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "counter_right_main_group/top_geometry",
            "object_type": "base",
        },
        {
            "id": "fridge",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "fridge_main_group",
            "object_type": "articulation",
            "params": {"openable_joint_name": "fridge_door_joint"},
        },
    ],
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

    def test_returns_dict_when_pass1_invalid(self, agent):
        invalid = dict(_MINIMAL_SPEC)
        invalid["embodiment"]["registry_name"] = "not_a_real_asset"
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(invalid))
        result = agent.generate_spec(
            "p",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
        )
        assert isinstance(result, dict)
        assert result["embodiment"]["registry_name"] == "not_a_real_asset"
        assert agent.last_validation_traces
        assert any("registry_name" in line for line in agent.last_validation_traces)

    @patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.load_usd_prim_tree")
    @patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.resolve_asset_usd_path")
    def test_two_pass_generate_spec_resolves_object_references(self, mock_resolve_usd, mock_load_tree, agent):
        mock_resolve_usd.return_value = "/tmp/scene.usd"
        mock_load_tree.return_value = _KITCHEN_PRIM_TREE
        agent.client.chat.completions.create.side_effect = [
            _chat_response(content=json.dumps(_KITCHEN_PASS1)),
            _chat_response(content=json.dumps(_RESOLVE_RESPONSE)),
        ]
        result = agent.generate_spec(
            "kitchen task",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
        )
        assert isinstance(result, ArenaEnvGraphSpec)
        assert agent.client.chat.completions.create.call_count == 2
        counter = next(ref for ref in result.object_references if ref.id == "counter_top")
        assert counter.prim_path == "counter_right_main_group/top_geometry"


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
    spec = agent.generate_spec(_ATOMIC_PICK_AND_PLACE_PROMPT)
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.last_validation_traces}"
    _assert_atomic_pick_and_place_spec(spec)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_five_bananas_parallel_pick_and_place_against_live_endpoint():
    """Live test: five bananas into one bin yields a parallel composite task."""
    agent = EnvironmentGenerationAgent()
    spec = agent.generate_spec(_FIVE_BANANAS_PROMPT)
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.last_validation_traces}"
    _assert_five_bananas_parallel_pick_and_place_spec(spec)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_resolve_usd_prim_robocasa_kitchen_counter_and_fridge():
    """End-to-end pass-1 + pass-2 prim resolution for Robocasa kitchen counter and fridge."""
    from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
    from isaaclab_arena.utils.asset_usd import resolve_asset_usd_path
    from isaaclab_arena.utils.usd_prim_tree import load_usd_prim_tree

    agent = EnvironmentGenerationAgent()
    asset_catalog = _catalog(
        "EMBODIMENTS:\n- droid_abs_joint_pos  tags=[default]\n\n"
        "BACKGROUNDS: lightwheel_robocasa_kitchen\n\n"
        "OBJECTS:\n"
        "- avocado01_fruits_veggies_robolab  tags=[]\n"
        "- plate_large_vomp_robolab  tags=[]\n"
        "- broccoli  tags=[]\n"
        "- sweet_potato  tags=[]"
    )
    task_catalog = _task_catalog(
        "TASKS (2):\n"
        "- PickAndPlaceTask (pick_up_object, destination_location, background_scene): Pick and place.\n"
        "- OpenDoorTask (openable_object): Open a door."
    )
    prompt = (
        "droid picks up an avocado on the counter top and places it in a plate; "
        "other veggies on the counter as distractors; then open the fridge door."
    )
    result = agent.generate_spec(
        prompt,
        asset_catalog=asset_catalog,
        task_catalog=task_catalog,
    )
    assert isinstance(result, ArenaEnvGraphSpec), f"spec validation failed: {agent.last_validation_traces}"
    assert result.object_references, "expected object_references for counter and fridge"

    usd_path = resolve_asset_usd_path(
        AssetSpec(
            id=result.background.id,
            registry_name=result.background.registry_name,
            params=result.background.params,
        ),
    )
    prim_paths = {record.relative_path for record in load_usd_prim_tree(usd_path)}

    counter_ref = next(
        (ref for ref in result.object_references if ref.object_type.value == "base"),
        None,
    )
    assert counter_ref is not None, "expected a base object_reference for the counter anchor"
    assert counter_ref.prim_path in prim_paths, f"counter prim_path not in USD tree: {counter_ref.prim_path!r}"
    assert "top_geometry" in (counter_ref.prim_path or ""), "counter anchor should reference a top_geometry prim"

    fridge_ref = next(
        (ref for ref in result.object_references if ref.object_type.value == "articulation"),
        None,
    )
    assert fridge_ref is not None, "expected an articulation object_reference for the fridge"
    assert fridge_ref.prim_path in prim_paths, f"fridge prim_path not in USD tree: {fridge_ref.prim_path!r}"
    assert "fridge_main_group" in (fridge_ref.prim_path or "")
    assert fridge_ref.params.get("openable_joint_name"), "fridge ref needs openable_joint_name"

    anchor = next(rel for rel in result.relations if rel.kind == "is_anchor")
    assert anchor.subject == counter_ref.id
    assert anchor.subject != result.background.id
