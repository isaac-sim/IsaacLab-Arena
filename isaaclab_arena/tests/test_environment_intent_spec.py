# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import pytest
from pydantic import ValidationError

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry


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


def test_environment_intent_spec_accepts_registered_relation_kind():
    registered = ObjectRelationLibraryRegistry().get_all_keys()
    assert "on" in registered
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
        "tasks": [],
    }
    spec = EnvironmentIntentSpec.model_validate(payload)
    assert spec.initial_state_graph[0].kind == "on"
