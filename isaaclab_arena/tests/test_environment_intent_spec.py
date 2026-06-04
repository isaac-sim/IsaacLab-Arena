# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import pytest
from pydantic import ValidationError

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    agent_ready_task_names,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import (
    EnvironmentIntentSpec,
    required_task_init_param_names,
)
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
from isaaclab_arena.tasks.close_door_task import CloseDoorTask
from isaaclab_arena.tasks.open_door_task import OpenDoorTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask


def test_required_task_init_param_names_match_task_constructors():
    assert required_task_init_param_names(PickAndPlaceTask) == [
        "pick_up_object",
        "destination_location",
        "background_scene",
    ]
    assert required_task_init_param_names(OpenDoorTask) == ["openable_object"]
    assert required_task_init_param_names(CloseDoorTask) == ["openable_object"]


def test_task_catalogue_lists_required_init_params():
    catalogue = build_task_catalogue()
    by_name = {entry.name: entry for entry in catalogue.tasks}
    assert by_name["PickAndPlaceTask"].required_params == [
        "pick_up_object",
        "destination_location",
        "background_scene",
    ]
    assert by_name["OpenDoorTask"].required_params == ["openable_object"]


def test_task_catalogue_lists_only_agent_ready_tasks():
    catalogue = build_task_catalogue()
    assert {entry.name for entry in catalogue.tasks} == agent_ready_task_names()


def test_relation_catalogue_matches_object_relation_registry():
    catalogue = build_relation_catalogue()
    registered = set(ObjectRelationLibraryRegistry().get_all_keys())
    catalogue_names = {entry.name for entry in catalogue.relations}
    assert catalogue_names == registered


def test_environment_intent_spec_rejects_missing_task_params():
    payload = {
        "reasoning": "test",
        "background": "kitchen",
        "embodiment": "franka_ik",
        "items": [],
        "initial_state_graph": [],
        "tasks": [{
            "kind": "PickAndPlaceTask",
            "params": {"pick_up_object": "cube"},
            "description": "pick and place",
        }],
    }
    with pytest.raises(ValidationError, match="missing required param"):
        EnvironmentIntentSpec.model_validate(payload)


def test_environment_intent_spec_rejects_non_agent_ready_task():
    payload = {
        "reasoning": "test",
        "background": "kitchen",
        "embodiment": "franka_ik",
        "items": [],
        "initial_state_graph": [],
        "tasks": [{
            "kind": "RotateRevoluteJointTask",
            "params": {},
            "description": "rotate a joint",
        }],
    }
    with pytest.raises(ValidationError, match="not agent-ready"):
        EnvironmentIntentSpec.model_validate(payload)


def test_environment_intent_spec_accepts_valid_task_params():
    payload = {
        "reasoning": "test",
        "background": "kitchen",
        "embodiment": "franka_ik",
        "items": [{"query": "cube", "role": "foreground", "category_tags": []}],
        "initial_state_graph": [{
            "kind": "on",
            "subject": "cube",
            "parent": "kitchen",
            "params": {},
        }],
        "tasks": [{
            "kind": "PickAndPlaceTask",
            "params": {
                "pick_up_object": "cube",
                "destination_location": "bowl",
                "background_scene": "kitchen",
            },
            "description": "pick up the cube and place it in the bowl",
        }],
    }
    spec = EnvironmentIntentSpec.model_validate(payload)
    assert spec.tasks[0].params["pick_up_object"] == "cube"
