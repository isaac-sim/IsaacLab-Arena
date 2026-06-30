# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live tests for ObjectReferenceAgent against real registered USD assets."""

from __future__ import annotations

import os

import pytest
from openai import OpenAI

import isaaclab_arena.assets.background_library  # noqa: F401
import isaaclab_arena.assets.object_library  # noqa: F401
from isaaclab_arena.agentic_environment_generation.agents.object_reference_agent import ObjectReferenceAgent
from isaaclab_arena.agentic_environment_generation.background_physics_catalog import (
    PhysicsPrimEntry,
    list_physics_prim_entries,
    resolve_background_usd_path,
)
from isaaclab_arena.agentic_environment_generation.environment_generation_agent import DEFAULT_BASE_URL, DEFAULT_MODEL
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_graph_types import SpatialRelationSpec, TaskSpec
from isaaclab_arena.utils.usd_helpers import open_stage


def _live_agent() -> ObjectReferenceAgent:
    api_key = os.getenv("NV_API_KEY")
    assert api_key, "NV_API_KEY is required for the live ObjectReferenceAgent tests"
    return ObjectReferenceAgent(OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL), DEFAULT_MODEL)


def _real_entries_or_skip(asset_name: str) -> tuple[str, list[PhysicsPrimEntry]]:
    registry = AssetRegistry()
    usd_path = resolve_background_usd_path(registry, asset_name)
    try:
        return usd_path, list_physics_prim_entries(usd_path)
    except Exception as exc:  # noqa: BLE001 - skip preserves live-test portability when remote USDs are unavailable.
        pytest.skip(f"Could not open registered USD for {asset_name!r}: {exc}")


def _entry_for_existing_static_prim_or_skip(usd_path: str, prim_path: str) -> PhysicsPrimEntry:
    try:
        with open_stage(usd_path) as stage:
            if not stage.GetPrimAtPath(prim_path):
                pytest.skip(f"Registered USD does not contain expected prim {prim_path!r}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not inspect registered USD {usd_path!r}: {exc}")
    return PhysicsPrimEntry(usd_prim_path=prim_path, physics_kinds=frozenset({"base"}))


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_object_reference_agent_finds_robocasa_microwave_subprims():
    """The live model should pick the actual Robocasa microwave articulation and disc paths."""
    usd_path, entries = _real_entries_or_skip("lightwheel_robocasa_kitchen")
    counter_entry = _entry_for_existing_static_prim_or_skip(usd_path, "/world/counter_right_main_group/top_geometry")
    microwave_entries = [entry for entry in entries if "microwave_main_group" in entry.usd_prim_path]
    candidate_entries = microwave_entries + [counter_entry]
    assert microwave_entries, "registered Robocasa kitchen USD must expose microwave physics prims"
    intent = EnvironmentIntentSpec(
        reasoning="open microwave, place avocado on the microwave plate, close microwave",
        background="lightwheel_robocasa_kitchen",
        embodiment="franka_ik",
        items=[
            Item(query="avocado", category_tags=["fruit"]),
            Item(query="microwave", category_tags=["appliance"]),
        ],
        initial_state_graph=[
            SpatialRelationSpec(kind="is_anchor", subject="kitchen_counter_top"),
            SpatialRelationSpec(kind="on", subject="avocado", reference="kitchen_counter_top"),
        ],
        tasks=[
            TaskSpec(kind="OpenDoorTask", params={"openable_object": "microwave"}, description="open microwave"),
            TaskSpec(
                kind="PickAndPlaceTask",
                params={
                    "pick_up_object": "avocado",
                    "destination_location": "microwave_plate",
                    "background_scene": "lightwheel_robocasa_kitchen",
                },
                description="place avocado on the microwave plate",
            ),
            TaskSpec(kind="CloseDoorTask", params={"openable_object": "microwave"}, description="close microwave"),
        ],
    )

    inference, _raw = _live_agent().infer_references(
        intent=intent,
        scope="background",
        parent_node_id="lightwheel_robocasa_kitchen",
        parent_asset_name="lightwheel_robocasa_kitchen",
        physics_entries=candidate_entries,
        usd_path=usd_path,
        reference_prompt=(
            "Return the microwave articulation as the openable reference and the microwave rotating disc "
            "as the rigid placement target. Also return the kitchen counter top geometry as the kitchen_counter_top "
            "base anchor reference."
        ),
        temperature=0.0,
    )
    refs_by_path = {ref.usd_prim_path: ref for ref in inference.object_references}

    assert "/world/microwave_main_group" in refs_by_path
    assert refs_by_path["/world/microwave_main_group"].object_type == "articulation"
    assert refs_by_path["/world/microwave_main_group"].openable_joint_name in {"microjoint", "Disc001_joint"}
    assert "/world/microwave_main_group/Microwave011_Disc001" in refs_by_path
    assert refs_by_path["/world/microwave_main_group/Microwave011_Disc001"].object_type == "rigid"
    assert "/world/counter_right_main_group/top_geometry" in refs_by_path
    assert refs_by_path["/world/counter_right_main_group/top_geometry"].id == "kitchen_counter_top"
    assert refs_by_path["/world/counter_right_main_group/top_geometry"].object_type == "base"


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_object_reference_agent_finds_maple_table_subprim():
    """The live model should pick the table subprim from the registered maple-table USD."""
    usd_path, entries = _real_entries_or_skip("maple_table_robolab")
    table_entries = [entry for entry in entries if entry.usd_prim_path.endswith("/table")]
    assert table_entries, "registered maple table USD must expose a /table physics prim"
    intent = EnvironmentIntentSpec(
        reasoning="use the maple table as the anchor surface",
        background="maple_table_robolab",
        embodiment="franka_ik",
        items=[Item(query="avocado", category_tags=["fruit"])],
        initial_state_graph=[SpatialRelationSpec(kind="is_anchor", subject="maple_table_robolab_table")],
        tasks=[],
    )

    inference, _raw = _live_agent().infer_references(
        intent=intent,
        scope="background",
        parent_node_id="maple_table_robolab",
        parent_asset_name="maple_table_robolab",
        physics_entries=table_entries,
        usd_path=usd_path,
        reference_prompt="Return the table surface as a rigid anchor reference with id maple_table_robolab_table.",
        temperature=0.0,
    )

    assert any(ref.usd_prim_path == table_entries[0].usd_prim_path for ref in inference.object_references)
    table_ref = next(ref for ref in inference.object_references if ref.usd_prim_path == table_entries[0].usd_prim_path)
    assert table_ref.id == "maple_table_robolab_table"
    assert table_ref.object_type == "rigid"
