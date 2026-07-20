# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import EnvironmentGenerationAgent
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import TaskCompositionType
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    catalog,
    chat_response,
    kitchen_pass1_dict,
    kitchen_prim_tree,
    kitchen_resolve_response,
    minimal_spec_dict,
    relation_catalog,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import task_catalog as make_task_catalog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(stub_openai):
    """A constructed ``EnvironmentGenerationAgent`` with a fully mocked openai client."""
    _, client = stub_openai
    a = EnvironmentGenerationAgent(api_key="test-key")
    client.chat.completions.create.reset_mock()
    return a, client


# ---------------------------------------------------------------------------
# generate_spec
# ---------------------------------------------------------------------------


class TestGenerateSpec:
    def test_builds_catalogues_from_singleton_registries_when_none(self, agent):
        agent_obj, client = agent
        client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_spec_dict()))
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
            mock_build_assets.return_value = catalog("<<ASSET-CATALOG>>")
            mock_build_relations.return_value = relation_catalog("<<RELATION-CATALOG>>")
            mock_build_tasks.return_value = make_task_catalog("<<TASK-CATALOG>>")
            agent_obj.generate_spec("p")
        mock_build_assets.assert_called_once_with()
        mock_build_relations.assert_called_once_with()
        mock_build_tasks.assert_called_once_with()

    @patch("isaaclab_arena.utils.usd_prim_tree.load_usd_prim_tree")
    @patch("isaaclab_arena.environment_spec.arena_env_graph_types.AssetSpec.resolve_usd_path")
    def test_two_pass_generate_spec_resolves_object_references(self, mock_resolve_usd, mock_load_tree, agent):
        agent_obj, client = agent
        mock_resolve_usd.return_value = "/tmp/scene.usd"
        mock_load_tree.return_value = kitchen_prim_tree()
        client.chat.completions.create.side_effect = [
            chat_response(content=json.dumps(kitchen_pass1_dict())),
            chat_response(content=json.dumps(kitchen_resolve_response())),
        ]
        spec, data = agent_obj.generate_spec(
            "kitchen task",
            asset_catalog=catalog("catalog"),
            relation_catalog=relation_catalog("RELATIONS"),
            task_catalog=make_task_catalog("TASKS"),
        )
        assert isinstance(spec, ArenaEnvGraphSpec)
        assert data is None
        assert client.chat.completions.create.call_count == 2
        assert spec.object_references

    @patch("isaaclab_arena.utils.usd_prim_tree.load_usd_prim_tree")
    @patch("isaaclab_arena.environment_spec.arena_env_graph_types.AssetSpec.resolve_usd_path")
    def test_two_pass_generate_spec_returns_dict_on_pass2_failure(self, mock_resolve_usd, mock_load_tree, agent):
        agent_obj, client = agent
        mock_resolve_usd.return_value = "/tmp/scene.usd"
        mock_load_tree.return_value = kitchen_prim_tree()
        bad_resolve = {
            "object_references": [{
                "id": "counter_top",
                "parent_id": "lightwheel_robocasa_kitchen",
                "prim_path": "missing_prim",
                "object_type": "base",
            }]
        }
        client.chat.completions.create.side_effect = [
            chat_response(content=json.dumps(kitchen_pass1_dict())),
            chat_response(content=json.dumps(bad_resolve)),
        ]
        spec, data = agent_obj.generate_spec(
            "kitchen task",
            asset_catalog=catalog("catalog"),
            relation_catalog=relation_catalog("RELATIONS"),
            task_catalog=make_task_catalog("TASKS"),
        )
        assert spec is None
        assert isinstance(data, dict)
        assert client.chat.completions.create.call_count == 2
        assert any("is not in the background prim tree" in line for line in agent_obj.traces)


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


# Marked flaky to absorb intermittent wire-level hiccups on the inference endpoint.
# TODO(qianl): drop the flaky marker once production-side retry is implemented.
@pytest.mark.skipif(not os.environ.get("NV_API_KEY"), reason="live endpoint test requires NV_API_KEY")
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_atomic_pick_and_place_against_live_endpoint():
    """Live test: avocado into bowl yields an atomic pick-and-place task."""
    agent = EnvironmentGenerationAgent()
    spec, data = agent.generate_spec(_ATOMIC_PICK_AND_PLACE_PROMPT)
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.traces}"
    assert data is None
    _assert_atomic_pick_and_place_spec(spec)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_five_bananas_parallel_pick_and_place_against_live_endpoint():
    """Live test: five bananas into one bin yields a parallel composite task."""
    agent = EnvironmentGenerationAgent()
    spec, data = agent.generate_spec(_FIVE_BANANAS_PROMPT)
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.traces}"
    assert data is None
    _assert_five_bananas_parallel_pick_and_place_spec(spec)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_resolve_usd_prim_robocasa_kitchen_counter_and_fridge():
    """End-to-end pass-1 + pass-2 prim resolution for Robocasa kitchen counter and fridge."""
    agent = EnvironmentGenerationAgent()
    asset_catalog = catalog(
        "EMBODIMENTS:\n- droid_abs_joint_pos  tags=[default]\n\n"
        "BACKGROUNDS: lightwheel_robocasa_kitchen\n\n"
        "OBJECTS:\n"
        "- avocado01_fruits_veggies_robolab  tags=[]\n"
        "- plate_large_vomp_robolab  tags=[]\n"
        "- broccoli  tags=[]\n"
        "- sweet_potato  tags=[]"
    )
    tasks = make_task_catalog(
        "TASKS (2):\n"
        "- PickAndPlaceTask (pick_up_object, destination_location, background_scene): Pick and place.\n"
        "- OpenDoorTask (openable_object): Open a door."
    )
    prompt = (
        "droid picks up an avocado on the counter top and places it in a plate; "
        "other veggies on the counter as distractors; then open the fridge door."
    )
    spec, data = agent.generate_spec(
        prompt,
        asset_catalog=asset_catalog,
        task_catalog=tasks,
    )
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.traces}"
    assert data is None
    assert spec.object_references, "expected object_references for counter and fridge"

    counter_ref = next(
        (ref for ref in spec.object_references if ref.object_type.value == "base"),
        None,
    )
    assert counter_ref is not None, "expected a base object_reference for the counter anchor"

    fridge_ref = next(
        (ref for ref in spec.object_references if ref.object_type.value == "articulation"),
        None,
    )
    assert fridge_ref is not None, "expected an articulation object_reference for the fridge"
    assert fridge_ref.params.get("openable_joint_name"), "fridge ref needs openable_joint_name"

    anchor = next(rel for rel in spec.relations if rel.kind == "is_anchor")
    assert anchor.subject == counter_ref.id
    assert anchor.subject != spec.background.id
