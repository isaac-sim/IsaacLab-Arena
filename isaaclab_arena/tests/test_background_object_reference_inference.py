# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaaclab_arena.agentic_environment_generation import intent_compiler as compiler_module
from isaaclab_arena.agentic_environment_generation.background_object_reference_spec import (
    BackgroundObjectReferenceInferenceSpec,
    BackgroundObjectReferenceItem,
    TaskParamBinding,
    validate_background_object_reference_inference,
)
from isaaclab_arena.agentic_environment_generation.background_object_reference_utils import (
    apply_background_object_reference_inference,
)
from isaaclab_arena.agentic_environment_generation.background_physics_catalog import PhysicsPrimEntry
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeType, SpatialRelationSpec, TaskSpec
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tests._asset_matcher_test_helpers import FakeAsset, make_registry


def _physics_entries() -> list[PhysicsPrimEntry]:
    return [
        PhysicsPrimEntry(
            usd_prim_path="/world/microwave_main_group",
            physics_kinds=frozenset({"articulation"}),
            revolute_joint_names=frozenset({"door_joint"}),
        ),
        PhysicsPrimEntry(
            usd_prim_path="/world/microwave_main_group/Microwave011_Disc001",
            physics_kinds=frozenset({"rigid_body"}),
        ),
    ]


def _open_pick_close_intent() -> EnvironmentIntentSpec:
    return EnvironmentIntentSpec(
        reasoning="open microwave, place avocado inside, close microwave",
        background="kitchen",
        embodiment="franka_ik",
        items=[
            Item(query="avocado", category_tags=["fruit"]),
            Item(query="microwave", category_tags=["appliance"]),
        ],
        initial_state_graph=[
            SpatialRelationSpec(kind="is_anchor", subject="kitchen"),
            SpatialRelationSpec(kind="on", subject="avocado", reference="kitchen"),
            SpatialRelationSpec(kind="on", subject="microwave", reference="kitchen"),
        ],
        tasks=[
            TaskSpec(kind="OpenDoorTask", params={"openable_object": "microwave"}, description="open microwave"),
            TaskSpec(
                kind="PickAndPlaceTask",
                params={
                    "pick_up_object": "avocado",
                    "destination_location": "microwave_plate",
                    "background_scene": "kitchen",
                },
                description="place avocado on microwave plate",
            ),
            TaskSpec(kind="CloseDoorTask", params={"openable_object": "microwave"}, description="close microwave"),
        ],
    )


def _valid_inference() -> BackgroundObjectReferenceInferenceSpec:
    return BackgroundObjectReferenceInferenceSpec(
        reasoning="microwave is built into kitchen",
        object_references=[
            BackgroundObjectReferenceItem(
                id="microwave_door",
                name="microwave_door",
                usd_prim_path="{ENV_REGEX_NS}/kitchen/microwave_door",
                object_type="articulation",
                openable_joint_name="door_joint",
            ),
            BackgroundObjectReferenceItem(
                id="microwave_plate",
                name="microwave_plate",
                usd_prim_path="{ENV_REGEX_NS}/kitchen/microwave_plate",
                object_type="rigid",
            ),
        ],
        task_param_bindings=[
            TaskParamBinding(task_index=0, param_name="openable_object", reference_id="microwave_door"),
            TaskParamBinding(task_index=1, param_name="destination_location", reference_id="microwave_plate"),
            TaskParamBinding(task_index=1, param_name="destination_object", reference_id="microwave_plate"),
            TaskParamBinding(task_index=2, param_name="openable_object", reference_id="microwave_door"),
        ],
        remove_item_ids=["microwave"],
    )


def test_reference_inference_validation_accepts_open_pick_close():
    inference = _valid_inference().model_copy(
        update={
            "object_references": [
                _valid_inference()
                .object_references[0]
                .model_copy(update={"usd_prim_path": "/world/microwave_main_group"}),
                _valid_inference()
                .object_references[1]
                .model_copy(update={"usd_prim_path": "/world/microwave_main_group/Microwave011_Disc001"}),
            ]
        }
    )
    validate_background_object_reference_inference(_open_pick_close_intent(), inference, _physics_entries())


def test_reference_inference_rejects_openable_without_joint():
    inference = _valid_inference().model_copy(
        update={
            "object_references": [
                _valid_inference()
                .object_references[0]
                .model_copy(update={"usd_prim_path": "/world/microwave_main_group", "openable_joint_name": None}),
                _valid_inference()
                .object_references[1]
                .model_copy(update={"usd_prim_path": "/world/microwave_main_group/Microwave011_Disc001"}),
            ]
        }
    )
    with pytest.raises(AssertionError, match="requires openable_joint_name"):
        validate_background_object_reference_inference(_open_pick_close_intent(), inference, _physics_entries())


def test_apply_reference_inference_removes_duplicate_item_and_rewrites_tasks():
    merged = apply_background_object_reference_inference(_open_pick_close_intent(), _valid_inference())

    assert [item.query for item in merged.items] == ["avocado"]
    assert all(rel.subject != "microwave" for rel in merged.initial_state_graph)
    assert merged.tasks[0].params["openable_object"] == "microwave_door"
    assert merged.tasks[1].params["destination_location"] == "microwave_plate"
    assert merged.tasks[1].params["destination_object"] == "microwave_plate"
    assert merged.tasks[2].params["openable_object"] == "microwave_door"


def test_compiler_emits_background_reference_nodes(monkeypatch):
    monkeypatch.setattr(compiler_module, "resolve_background_usd_path", lambda *_: "/tmp/unused.usd")
    registry = make_registry([
        FakeAsset(name="kitchen", tags=["background"]),
        FakeAsset(name="franka_ik", tags=["embodiment", "ik"]),
        FakeAsset(name="avocado01_fruits_robolab", tags=["object", "fruit"]),
    ])
    merged = apply_background_object_reference_inference(_open_pick_close_intent(), _valid_inference())

    graph = IntentCompiler(registry=registry).compile(merged)

    node_ids = [node.id for node in graph.nodes]
    assert node_ids[:3] == ["kitchen", "microwave_door", "microwave_plate"]
    assert graph.nodes_by_id["microwave_door"].type == ArenaEnvGraphNodeType.OBJECT_REFERENCE
    assert graph.nodes_by_id["microwave_door"].object_type == ObjectType.ARTICULATION
    assert graph.nodes_by_id["microwave_door"].params["openable_joint_name"] == "door_joint"
    assert graph.nodes_by_id["microwave_plate"].object_type == ObjectType.RIGID
    assert graph.tasks[1].params["destination_object"] == "microwave_plate"


def test_compiler_uses_object_references_from_intent_field(monkeypatch):
    monkeypatch.setattr(compiler_module, "resolve_background_usd_path", lambda *_: "/tmp/unused.usd")
    registry = make_registry([
        FakeAsset(name="kitchen", tags=["background"]),
        FakeAsset(name="franka_ik", tags=["embodiment", "ik"]),
        FakeAsset(name="avocado01_fruits_robolab", tags=["object", "fruit"]),
    ])
    merged = apply_background_object_reference_inference(_open_pick_close_intent(), _valid_inference())

    graph = IntentCompiler(registry=registry).compile(merged)

    assert "microwave_door" in graph.nodes_by_id
    assert "microwave_plate" in graph.nodes_by_id
    assert graph.tasks[0].params["openable_object"] == "microwave_door"


def test_pick_and_place_mimic_cfg_falls_back_to_destination_location():
    task = PickAndPlaceTask.__new__(PickAndPlaceTask)
    task.mimic_env_cfg_factory = None
    task.pick_up_object = type("Obj", (), {"name": "avocado"})()
    task.destination_object = None
    task.destination_location = type("Obj", (), {"name": "microwave_plate"})()

    cfg = PickAndPlaceTask.get_mimic_env_cfg(task, ArmMode.SINGLE_ARM)

    assert cfg.pick_up_object_name == "avocado"
    assert cfg.destination_location_name == "microwave_plate"
