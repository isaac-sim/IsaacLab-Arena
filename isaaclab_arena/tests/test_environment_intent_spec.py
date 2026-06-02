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
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry, TaskRegistry


def test_task_catalogue_lists_only_agent_ready_tasks():
    catalogue = build_task_catalogue()
    assert {entry.name for entry in catalogue.tasks} == agent_ready_task_names()
    assert "PickAndPlaceTask" in {entry.name for entry in catalogue.tasks}
    assert "OpenDoorTask" in {entry.name for entry in catalogue.tasks}
    assert "CloseDoorTask" in {entry.name for entry in catalogue.tasks}


def test_task_catalogue_names_match_task_registry_keys():
    catalogue = build_task_catalogue()
    registered = set(TaskRegistry().get_all_keys())
    for entry in catalogue.tasks:
        assert entry.name in registered
        assert TaskRegistry().get_task_by_name(entry.name) is not None


def test_relation_catalogue_matches_object_relation_registry():
    catalogue = build_relation_catalogue()
    registered = set(ObjectRelationLibraryRegistry().get_all_keys())
    catalogue_names = {entry.name for entry in catalogue.relations}
    assert catalogue_names == registered


def test_environment_intent_spec_rejects_unregistered_relation_kind():
    payload = {
        "reasoning": "test",
        "task_description": "test",
        "background": "kitchen",
        "embodiment": "franka_ik",
        "items": [],
        "initial_state_graph": [{
            "kind": "not_a_real_relation",
            "subject": "cube",
            "target": "table",
            "params": {},
        }],
        "tasks": [],
    }
    with pytest.raises(ValidationError, match="not registered"):
        EnvironmentIntentSpec.model_validate(payload)


def test_environment_intent_spec_rejects_non_agent_ready_task():
    payload = {
        "reasoning": "test",
        "task_description": "test",
        "background": "kitchen",
        "embodiment": "franka_ik",
        "items": [],
        "initial_state_graph": [],
        "tasks": [{
            "kind": "SortingTask",
            "subject": "cube",
            "target": None,
            "description": "sort objects",
        }],
    }
    with pytest.raises(ValidationError, match="not agent-ready"):
        EnvironmentIntentSpec.model_validate(payload)


def test_environment_intent_spec_accepts_registered_relation_and_task_names():
    registered_relations = ObjectRelationLibraryRegistry().get_all_keys()
    assert "on" in registered_relations
    assert "PickAndPlaceTask" in agent_ready_task_names()
    payload = {
        "reasoning": "test",
        "task_description": "test",
        "background": "kitchen",
        "embodiment": "franka_ik",
        "items": [{"query": "cube", "role": "foreground", "category_tags": []}],
        "initial_state_graph": [{
            "kind": "on",
            "subject": "cube",
            "target": "kitchen",
            "params": {},
        }],
        "tasks": [{
            "kind": "PickAndPlaceTask",
            "subject": "cube",
            "target": "bowl",
            "description": "pick up the cube and place it in the bowl",
        }],
    }
    spec = EnvironmentIntentSpec.model_validate(payload)
    assert spec.initial_state_graph[0].kind == "on"
    assert spec.tasks[0].kind == "PickAndPlaceTask"
