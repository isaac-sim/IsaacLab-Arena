# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    EnvironmentGenerationAgent,
    build_asset_catalogue,
)
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadyCandidateCatalogue,
    SimReadyObjectCandidate,
)
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.simready_constants import SIMREADY_USD_OBJECT_REGISTRY_NAME
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import TaskCompositionType
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    catalogs,
    kitchen_pass1_dict,
    kitchen_prim_tree,
    kitchen_resolve_response,
    minimal_spec_dict,
    skip_without_live_endpoint_key,
    stub_responses,
)

_AGENT_MODULE = "isaaclab_arena.agentic_environment_generation.environment_generation_agent"

_TRASH_CAN = SimReadyObjectCandidate(
    search_phrase="green trash can",
    usd_path="s3://bucket/trash_can.usd",
    tags=("sim-ready", "green", "trash", "can"),
)
"""The one SimReady hit the search is stubbed to return."""


@pytest.fixture
def agent(stub_openai):
    """A constructed ``EnvironmentGenerationAgent`` with a fully mocked openai client."""
    _, client = stub_openai
    a = EnvironmentGenerationAgent(api_key="test-key")
    client.chat.completions.create.reset_mock()
    return a, client


@pytest.fixture
def simready_agent(agent):
    """An agent with the SimReady search on, yielding ``(agent, client, search_mock)``.

    The search is stubbed to find ``_TRASH_CAN``; a test that needs another outcome reassigns
    the mock's return value.
    """
    agent_obj, client = agent
    agent_obj.enable_simready_search = True
    with patch(f"{_AGENT_MODULE}.search_simready_objects") as mock_search:
        mock_search.return_value = SimReadyCandidateCatalogue(candidates=[_TRASH_CAN])
        yield agent_obj, client, mock_search


def _missing_objects(*phrases: str) -> dict:
    """The reply from the pass that names what the asset catalog does not cover."""
    return {"search_phrases": list(phrases)}


class TestGenerateSpec:
    def test_builds_catalogues_from_singleton_registries_when_none(self, agent):
        agent_obj, client = agent
        stub_responses(client, minimal_spec_dict())
        with (
            patch(f"{_AGENT_MODULE}.build_asset_catalogue") as mock_build_assets,
            patch(f"{_AGENT_MODULE}.build_relation_catalogue") as mock_build_relations,
            patch(f"{_AGENT_MODULE}.build_task_catalogue") as mock_build_tasks,
        ):
            built = catalogs("<<ASSET-CATALOG>>", "<<RELATION-CATALOG>>", "<<TASK-CATALOG>>")
            mock_build_assets.return_value = built["asset_catalog"]
            mock_build_relations.return_value = built["relation_catalog"]
            mock_build_tasks.return_value = built["task_catalog"]
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
        stub_responses(client, kitchen_pass1_dict(), kitchen_resolve_response())
        spec, data = agent_obj.generate_spec("kitchen task", **catalogs())
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
            "object_references": [{"id": "counter_top", "prim_path": "missing_prim", "openable_joint_name": None}]
        }
        stub_responses(client, kitchen_pass1_dict(), bad_resolve)
        spec, data = agent_obj.generate_spec("kitchen task", **catalogs())
        assert spec is None
        assert isinstance(data, dict)
        assert client.chat.completions.create.call_count == 2
        assert any("is not in the background prim tree" in line for line in agent_obj.traces)

    @staticmethod
    def _generate_with_simready(agent_obj):
        return agent_obj.generate_spec("p", **catalogs())

    def test_generate_spec_catalogues_what_the_search_finds_before_inferring_the_spec(self, simready_agent):
        agent_obj, client, mock_search = simready_agent
        stub_responses(client, _missing_objects("green trash can"), minimal_spec_dict())
        spec, data = self._generate_with_simready(agent_obj)
        assert isinstance(spec, ArenaEnvGraphSpec)
        assert data is None
        assert mock_search.call_args.args[0] == ["green trash can"]
        # The found asset reaches spec inference as an ordinary OBJECTS entry, so the model picks
        # it by name and the spec needs no usd_path of its own.
        spec_user_msg = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "simready_green_trash_can" in spec_user_msg
        assert _TRASH_CAN.usd_path not in spec_user_msg
        assert AssetRegistry().is_registered("simready_green_trash_can")

    def test_a_searched_simready_object_reaches_the_spec_with_its_usd_path(self, simready_agent):
        agent_obj, client, _ = simready_agent
        spec_dict = minimal_spec_dict()
        spec_dict["objects"].append({
            "id": "trash_can",
            "registry_name": "simready_green_trash_can",
            "params": {},
        })
        stub_responses(client, _missing_objects("green trash can"), spec_dict)
        spec, _ = self._generate_with_simready(agent_obj)
        assert isinstance(spec, ArenaEnvGraphSpec)
        searched = next(asset for asset in spec.objects if asset.id == "trash_can")
        # The search name only exists in the process that searched, so the object is rewritten onto
        # the generic SimReady asset every process has, carrying the path the search found.
        assert searched.registry_name == SIMREADY_USD_OBJECT_REGISTRY_NAME
        assert searched.params == {"usd_path": _TRASH_CAN.usd_path}
        # Tags in params would be a duplicate keyword argument at build time.
        assert all("tags" not in asset["params"] for asset in spec.to_dict()["objects"])

    def test_a_searched_simready_object_in_an_object_set_is_rejected(self, simready_agent):
        agent_obj, client, _ = simready_agent
        spec_dict = minimal_spec_dict()
        spec_dict["object_sets"] = [{"id": "bins", "members": ["simready_green_trash_can"]}]
        stub_responses(client, _missing_objects("green trash can"), spec_dict)
        spec, data = self._generate_with_simready(agent_obj)
        # A member is a bare registered name with nowhere to carry a usd_path, so the set would
        # name a search entry that exists in no other process.
        assert spec is None
        assert isinstance(data, dict)
        assert any("nowhere to carry a usd_path" in trace for trace in agent_obj.traces)

    def test_a_catalogue_object_keeps_its_own_registry_name(self, simready_agent):
        agent_obj, client, _ = simready_agent
        stub_responses(client, _missing_objects("green trash can"), minimal_spec_dict())
        # The model picked catalogue objects instead, and those resolve by name in any process.
        spec, _ = self._generate_with_simready(agent_obj)
        assert all(asset.registry_name != SIMREADY_USD_OBJECT_REGISTRY_NAME for asset in spec.objects)
        assert all("usd_path" not in asset.params for asset in spec.objects)
        assert "simready_assets" not in spec.to_dict()

    def test_generate_spec_builds_the_spec_without_an_object_no_asset_was_found_for(self, simready_agent):
        agent_obj, client, mock_search = simready_agent
        mock_search.return_value = SimReadyCandidateCatalogue(unmatched_phrases=["green trash can"])
        stub_responses(client, _missing_objects("green trash can"), minimal_spec_dict())
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

    def test_generate_spec_skips_the_search_when_the_catalog_covers_the_prompt(self, simready_agent):
        agent_obj, client, mock_search = simready_agent
        stub_responses(client, _missing_objects(), minimal_spec_dict())
        spec, _ = self._generate_with_simready(agent_obj)
        assert isinstance(spec, ArenaEnvGraphSpec)
        mock_search.assert_not_called()
        assert agent_obj.unavailable_objects == ()

    def test_generate_spec_leaves_the_inference_path_alone_when_the_search_is_off(self, simready_agent):
        agent_obj, client, mock_search = simready_agent
        agent_obj.enable_simready_search = False
        stub_responses(client, minimal_spec_dict())
        spec, _ = agent_obj.generate_spec("p", **catalogs())
        assert isinstance(spec, ArenaEnvGraphSpec)
        # One call and one only: no pass asking what the catalog misses.
        assert client.chat.completions.create.call_count == 1
        mock_search.assert_not_called()


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


# The tests below call the inference endpoint for real. They are marked flaky to absorb
# intermittent wire-level hiccups.
# TODO(qianl): drop the flaky marker once production-side retry is implemented.


def _generate_live(prompt: str, **catalog_kwargs) -> ArenaEnvGraphSpec:
    """Generate a spec from ``prompt`` against the live endpoint, asserting it validated."""
    agent = EnvironmentGenerationAgent()
    spec, data = agent.generate_spec(prompt, **catalog_kwargs)
    assert isinstance(spec, ArenaEnvGraphSpec), f"spec validation failed: {agent.traces}"
    assert data is None
    return spec


@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_atomic_pick_and_place_against_live_endpoint():
    """Live test: avocado into bowl yields an atomic pick-and-place task."""
    spec = _generate_live("Franka picks up the avocado and place it in the bowl on the maple table")

    assert len(spec.objects) == 2, f"expected 2 objects, got {len(spec.objects)}"
    # The maple table is the background itself, so it is what the scene anchors to. How many other
    # anchors the model adds is left open, as is whether the embodiment gets its own 'on' relation.
    anchors = {relation.subject for relation in spec.relations if relation.kind == "is_anchor"}
    assert spec.background.id in anchors, f"expected the background anchored, got {anchors}"
    on_subjects = {relation.subject for relation in spec.relations if relation.kind == "on"}
    assert {obj.id for obj in spec.objects} <= on_subjects, "every object needs an 'on' relation"

    assert spec.task.composition is TaskCompositionType.ATOMIC
    assert len(spec.task.subtasks) == 1
    assert spec.task.subtasks[0].kind == "PickAndPlaceTask"


@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_five_bananas_parallel_pick_and_place_against_live_endpoint():
    """Live test: five bananas into one bin yields a parallel composite task."""
    spec = _generate_live(
        "There are five bananas and a grey bin on the maple table. Droid places all the bananas into the bin."
    )

    assert len(spec.objects) == 6, f"expected 6 objects, got {len(spec.objects)}"
    on_subjects = {relation.subject for relation in spec.relations if relation.kind == "on"}
    assert {obj.id for obj in spec.objects} <= on_subjects, "every object needs an 'on' relation"

    assert spec.task.composition is TaskCompositionType.PARALLEL
    assert len(spec.task.subtasks) == 5
    assert {leaf.kind for leaf in spec.task.subtasks} == {"PickAndPlaceTask"}

    pick_ids = [leaf.params["pick_up_object"] for leaf in spec.task.subtasks]
    dest_ids = {leaf.params["destination_location"] for leaf in spec.task.subtasks}
    assert len(set(pick_ids)) == 5, f"expected 5 distinct pick objects, got {pick_ids!r}"
    assert len(dest_ids) == 1, f"expected one shared destination, got {dest_ids!r}"
    assert dest_ids.isdisjoint(pick_ids), f"destination {dest_ids!r} should not be among pick objects"


@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_resolve_usd_prim_robocasa_kitchen_counter_and_fridge():
    """End-to-end pass-1 + pass-2 prim resolution for Robocasa kitchen counter and fridge."""
    spec = _generate_live(
        "droid picks up an avocado on the counter top and places it in a plate; "
        "other veggies on the counter as distractors; then open the fridge door.",
        # The relation catalogue is left out so the real one is built.
        **catalogs(
            assets=(
                "EMBODIMENTS:\n- droid_abs_joint_pos  tags=[default]\n\n"
                "BACKGROUNDS: lightwheel_robocasa_kitchen\n\n"
                "OBJECTS:\n"
                "- avocado01_fruits_veggies_robolab  tags=[]\n"
                "- plate_large_vomp_robolab  tags=[]\n"
                "- broccoli  tags=[]\n"
                "- sweet_potato  tags=[]"
            ),
            relations=None,
            tasks=(
                "TASKS (2):\n"
                "- PickAndPlaceTask (pick_up_object, destination_location, background_scene): Pick and place.\n"
                "- OpenDoorTask (openable_object): Open a door."
            ),
        ),
    )
    # Which kitchen surfaces the model names is left open; that each one resolves to a prim, that
    # the opened one is an articulation with a joint, and that the anchors are surfaces rather than
    # the room is not.
    assert spec.object_references, "expected object_references for counter and fridge"
    assert all(ref.prim_path for ref in spec.object_references), "every object_reference needs a prim_path"

    opened_refs = [ref for ref in spec.object_references if ref.params.get("openable_joint_name")]
    assert opened_refs, "expected the fridge reference to carry an openable_joint_name"
    assert all(ref.object_type.value == "articulation" for ref in opened_refs)

    anchors = {rel.subject for rel in spec.relations if rel.kind == "is_anchor"}
    assert anchors & {ref.id for ref in spec.object_references}, f"expected a surface anchored, got {anchors}"
