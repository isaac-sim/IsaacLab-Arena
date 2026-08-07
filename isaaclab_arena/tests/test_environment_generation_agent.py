# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from enum import Enum
from unittest.mock import patch

import pytest

from isaaclab_arena.agentic_environment_generation.catalogues import (
    AssetCatalogue,
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.environment_generation_agent import EnvironmentGenerationAgent
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadyCandidateCatalogue,
    SimReadyObjectCandidate,
)
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.simready_constants import SIMREADY_USD_OBJECT_REGISTRY_NAME
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
    skip_without_live_endpoint_key,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import task_catalog as make_task_catalog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TestTaskMode(Enum):
    FAST = "fast"
    PRECISE = "precise"


class _TestTask:
    """Test task."""

    agent_ready = True

    def __init__(self, target, mode: _TestTaskMode = _TestTaskMode.FAST, retries: int = 1):
        pass


class _TestTaskRegistry:
    def get_all_keys(self):
        return ["TestTask"]

    def get_task_by_name(self, name):
        assert name == "TestTask"
        return _TestTask


@pytest.fixture
def agent(stub_openai):
    """A constructed ``EnvironmentGenerationAgent`` with a fully mocked openai client."""
    _, client = stub_openai
    a = EnvironmentGenerationAgent(api_key="test-key")
    client.chat.completions.create.reset_mock()
    return a, client


def test_task_catalogue_collects_required_optional_and_enum_params():
    catalogue = build_task_catalogue(_TestTaskRegistry())

    assert (
        catalogue.to_catalog_string()
        == "TASKS (1):\n- TestTask (required: target; optional: mode={fast, precise}, retries): Test task."
    )


def test_task_catalogue_excludes_structural_graph_fields():
    """Params already expressed on CompositeTaskSpec (e.g. description) stay out of TASKS."""
    catalogue = build_task_catalogue()
    entries = {entry.name: entry for entry in catalogue.tasks}

    assert "PickAndPlaceTask" in entries
    pick_params = entries["PickAndPlaceTask"].required_params + entries["PickAndPlaceTask"].optional_params
    assert "task_description" not in pick_params


def test_relation_catalogue_collects_required_optional_and_enum_params():
    catalogue = build_relation_catalogue()
    entries = {entry.name: entry for entry in catalogue.relations}

    assert set(entries) == {
        "is_anchor",
        "next_to",
        "not_next_to",
        "on",
        "rotate_around_solution",
    }
    assert entries["next_to"].required_params == ["side"]
    assert entries["next_to"].optional_params == [
        "relation_loss_weight",
        "distance_m",
        "cross_position_ratio",
        "tolerance_m",
    ]
    assert entries["next_to"].enum_options == {"side": ["positive_x", "negative_x", "positive_y", "negative_y"]}
    for entry in entries.values():
        assert "parent" not in entry.required_params + entry.optional_params


# ---------------------------------------------------------------------------
# generate_spec
# ---------------------------------------------------------------------------


class TestGenerateSpec:
    def test_builds_catalogues_from_singleton_registries_when_none(self, agent):
        agent_obj, client = agent
        client.chat.completions.create.side_effect = [
            chat_response(content=json.dumps(minimal_spec_dict())),
        ]
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

    @staticmethod
    def _missing_objects_response(*phrases: str):
        """A canned reply from the pass that names what the asset catalog does not cover."""
        return chat_response(content=json.dumps({"search_phrases": list(phrases)}))

    def _generate_with_simready(self, agent_obj, asset_catalog=None):
        agent_obj.enable_simready_search = True
        return agent_obj.generate_spec(
            "p",
            asset_catalog=asset_catalog or catalog("catalog"),
            relation_catalog=relation_catalog("RELATIONS"),
            task_catalog=make_task_catalog("TASKS"),
        )

    @patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.search_simready_objects")
    def test_generate_spec_catalogues_what_the_search_finds_before_inferring_the_spec(self, mock_search, agent):
        agent_obj, client = agent
        mock_search.return_value = SimReadyCandidateCatalogue(
            candidates=[
                SimReadyObjectCandidate(
                    search_phrase="green trash can",
                    usd_path="s3://bucket/trash_can.usd",
                    tags=("sim-ready", "green", "trash", "can"),
                )
            ]
        )
        client.chat.completions.create.side_effect = [
            self._missing_objects_response("green trash can"),
            chat_response(content=json.dumps(minimal_spec_dict())),
        ]
        spec, data = self._generate_with_simready(agent_obj)
        assert isinstance(spec, ArenaEnvGraphSpec)
        assert data is None
        assert mock_search.call_args.args[0] == ["green trash can"]
        # The found asset reaches spec inference as an ordinary OBJECTS entry, so the model picks
        # it by name and the spec needs no usd_path of its own.
        spec_user_msg = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "simready_green_trash_can" in spec_user_msg
        assert "s3://bucket/trash_can.usd" not in spec_user_msg
        assert AssetRegistry().is_registered("simready_green_trash_can")

    @patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.search_simready_objects")
    def test_a_searched_simready_object_reaches_the_spec_with_its_usd_path(self, mock_search, agent):
        agent_obj, client = agent
        mock_search.return_value = SimReadyCandidateCatalogue(
            candidates=[
                SimReadyObjectCandidate(
                    search_phrase="green trash can",
                    usd_path="s3://bucket/trash_can.usd",
                    tags=("sim-ready", "green", "trash", "can"),
                )
            ]
        )
        spec_dict = minimal_spec_dict()
        spec_dict["objects"].append({
            "id": "trash_can",
            "registry_name": "simready_green_trash_can",
            "params": {},
        })
        client.chat.completions.create.side_effect = [
            self._missing_objects_response("green trash can"),
            chat_response(content=json.dumps(spec_dict)),
        ]
        spec, _ = self._generate_with_simready(agent_obj)
        assert isinstance(spec, ArenaEnvGraphSpec)
        searched = next(asset for asset in spec.objects if asset.id == "trash_can")
        # The search name only exists in the process that searched, so the object is rewritten onto
        # the generic SimReady asset every process has, carrying the path the search found.
        assert searched.registry_name == SIMREADY_USD_OBJECT_REGISTRY_NAME
        assert searched.params == {"usd_path": "s3://bucket/trash_can.usd"}
        # Tags in params would be a duplicate keyword argument at build time.
        assert all("tags" not in asset["params"] for asset in spec.to_dict()["objects"])

    @patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.search_simready_objects")
    def test_a_searched_simready_object_in_an_object_set_is_rejected(self, mock_search, agent):
        agent_obj, client = agent
        mock_search.return_value = SimReadyCandidateCatalogue(
            candidates=[SimReadyObjectCandidate(search_phrase="green trash can", usd_path="s3://bucket/trash_can.usd")]
        )
        spec_dict = minimal_spec_dict()
        spec_dict["object_sets"] = [{"id": "bins", "members": ["simready_green_trash_can"]}]
        client.chat.completions.create.side_effect = [
            self._missing_objects_response("green trash can"),
            chat_response(content=json.dumps(spec_dict)),
        ]
        spec, data = self._generate_with_simready(agent_obj)
        # A member is a bare registered name with nowhere to carry a usd_path, so the set would
        # name a search entry that exists in no other process.
        assert spec is None
        assert isinstance(data, dict)
        assert any("nowhere to carry a usd_path" in trace for trace in agent_obj.traces)

    @patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.search_simready_objects")
    def test_a_catalogue_object_keeps_its_own_registry_name(self, mock_search, agent):
        agent_obj, client = agent
        mock_search.return_value = SimReadyCandidateCatalogue(
            candidates=[SimReadyObjectCandidate(search_phrase="green trash can", usd_path="s3://bucket/trash_can.usd")]
        )
        client.chat.completions.create.side_effect = [
            self._missing_objects_response("green trash can"),
            chat_response(content=json.dumps(minimal_spec_dict())),
        ]
        # The model picked catalogue objects instead, and those resolve by name in any process.
        spec, _ = self._generate_with_simready(agent_obj)
        assert all(asset.registry_name != SIMREADY_USD_OBJECT_REGISTRY_NAME for asset in spec.objects)
        assert all("usd_path" not in asset.params for asset in spec.objects)
        assert "simready_assets" not in spec.to_dict()

    @patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.search_simready_objects")
    def test_generate_spec_builds_the_spec_without_an_object_no_asset_was_found_for(self, mock_search, agent):
        agent_obj, client = agent
        mock_search.return_value = SimReadyCandidateCatalogue(unmatched_phrases=["green trash can"])
        client.chat.completions.create.side_effect = [
            self._missing_objects_response("green trash can"),
            chat_response(content=json.dumps(minimal_spec_dict())),
        ]
        spec, data = self._generate_with_simready(agent_obj)
        # Nothing to walk back: the object was never offered, so the first answer is usable.
        assert isinstance(spec, ArenaEnvGraphSpec)
        assert data is None
        assert client.chat.completions.create.call_count == 2
        # The caller is still told what the prompt asked for and could not get.
        assert agent_obj.unavailable_objects == ("green trash can",)
        # Nothing defeated the generation, so nothing lands in the error channel. A search that
        # came up short is reported structurally above and logged, not raised as a failure.
        assert agent_obj.traces == ()

    @patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.search_simready_objects")
    def test_generate_spec_skips_the_search_when_the_catalog_covers_the_prompt(self, mock_search, agent):
        agent_obj, client = agent
        client.chat.completions.create.side_effect = [
            self._missing_objects_response(),
            chat_response(content=json.dumps(minimal_spec_dict())),
        ]
        spec, _ = self._generate_with_simready(agent_obj)
        assert isinstance(spec, ArenaEnvGraphSpec)
        mock_search.assert_not_called()
        assert agent_obj.unavailable_objects == ()

    @patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.search_simready_objects")
    def test_generate_spec_leaves_the_inference_path_alone_when_the_search_is_off(self, mock_search, agent):
        agent_obj, client = agent
        client.chat.completions.create.side_effect = [
            chat_response(content=json.dumps(minimal_spec_dict())),
        ]
        spec, _ = agent_obj.generate_spec(
            "p",
            asset_catalog=catalog("catalog"),
            relation_catalog=relation_catalog("RELATIONS"),
            task_catalog=make_task_catalog("TASKS"),
        )
        assert isinstance(spec, ArenaEnvGraphSpec)
        # One call and one only: no pass asking what the catalog misses.
        assert client.chat.completions.create.call_count == 1
        mock_search.assert_not_called()


# ---------------------------------------------------------------------------
# Asset catalogue
# ---------------------------------------------------------------------------


def test_asset_catalogue_reports_object_type():
    catalog_string = build_asset_catalogue().to_catalog_string()

    assert "- banana_ycb_robolab  type=rigid" in catalog_string
    assert "- microwave  type=articulation" in catalog_string


def test_asset_catalogue_withholds_the_generic_simready_object():
    # It spawns only from a usd_path the model cannot know, so offering it is what used to put an
    # invented usd_path and tags in the generated spec.
    catalog_string = build_asset_catalogue().to_catalog_string()

    assert AssetRegistry().is_registered(SIMREADY_USD_OBJECT_REGISTRY_NAME)
    assert SIMREADY_USD_OBJECT_REGISTRY_NAME not in catalog_string


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
@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_atomic_pick_and_place_against_live_endpoint():
    """Live test: avocado into bowl yields an atomic pick-and-place task."""
    agent = EnvironmentGenerationAgent()
    spec, data = agent.generate_spec(_ATOMIC_PICK_AND_PLACE_PROMPT)
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.traces}"
    assert data is None
    _assert_atomic_pick_and_place_spec(spec)


@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_five_bananas_parallel_pick_and_place_against_live_endpoint():
    """Live test: five bananas into one bin yields a parallel composite task."""
    agent = EnvironmentGenerationAgent()
    spec, data = agent.generate_spec(_FIVE_BANANAS_PROMPT)
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.traces}"
    assert data is None
    _assert_five_bananas_parallel_pick_and_place_spec(spec)


@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_resolve_usd_prim_robocasa_kitchen_counter_and_fridge():
    """End-to-end pass-1 + pass-2 prim resolution for Robocasa kitchen counter and fridge."""
    agent = EnvironmentGenerationAgent()
    # Keep structured entries in sync with the prompt: catalog validation checks
    # AssetCatalogue.objects, not a to_catalog_string override.
    asset_catalog = AssetCatalogue(
        embodiments=[{"name": "droid_abs_joint_pos", "tags": ["default"]}],
        backgrounds=[{"name": "lightwheel_robocasa_kitchen", "tags": []}],
        objects=[
            {"name": "avocado01_fruits_veggies_robolab", "object_type": "rigid", "tags": []},
            {"name": "plate_large_vomp_robolab", "object_type": "rigid", "tags": []},
            {"name": "broccoli", "object_type": "rigid", "tags": []},
            {"name": "sweet_potato", "object_type": "rigid", "tags": []},
        ],
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
