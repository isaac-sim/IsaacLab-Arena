# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pass-2 object_reference prim_path resolution."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver import ObjectReferencePrimResolver
from isaaclab_arena.agentic_environment_generation.query_backend import QueryBackend
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    chat_response,
    kitchen_pass1_dict,
    kitchen_prim_tree,
    kitchen_resolve_response,
)


@patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.load_usd_prim_tree")
@patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.resolve_asset_usd_path")
def test_object_reference_prim_resolver_infer_merges_llm_output(mock_resolve_usd, mock_load_tree):
    mock_resolve_usd.return_value = "/tmp/scene.usd"
    mock_load_tree.return_value = kitchen_prim_tree()
    client = MagicMock()
    client.chat.completions.create.return_value = chat_response(json.dumps(kitchen_resolve_response()))
    resolver = ObjectReferencePrimResolver(QueryBackend(client, "test-model"))
    spec = ArenaEnvGraphSpec.model_validate(kitchen_pass1_dict())
    merged = resolver.infer(spec, [])
    client.chat.completions.create.assert_called_once()
    counter = next(ref for ref in merged.object_references if ref.id == "counter_top")
    fridge = next(ref for ref in merged.object_references if ref.id == "fridge")
    assert counter.prim_path == "counter_right_main_group/top_geometry"
    assert fridge.prim_path == "fridge_main_group"
    assert fridge.params["openable_joint_name"] == "fridge_door_joint"


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (
            {
                "object_references": [{
                    "id": "counter_top",
                    "parent_id": "lightwheel_robocasa_kitchen",
                    "prim_path": None,
                    "object_type": "base",
                }]
            },
            "requires a prim_path",
        ),
        (
            {
                "object_references": [{
                    "id": "counter_top",
                    "parent_id": "lightwheel_robocasa_kitchen",
                    "prim_path": "missing_prim",
                    "object_type": "base",
                }]
            },
            "is not in the background prim tree",
        ),
        (
            {
                "object_references": [{
                    "id": "counter_top",
                    "parent_id": "lightwheel_robocasa_kitchen",
                    "prim_path": "counter_right_main_group/top_geometry",
                    "object_type": "rigid",
                }]
            },
            "does not match prim tree",
        ),
    ],
)
@patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.load_usd_prim_tree")
@patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.resolve_asset_usd_path")
def test_object_reference_prim_resolver_infer_rejects_invalid_llm_output(
    mock_resolve_usd,
    mock_load_tree,
    response,
    match,
):
    mock_resolve_usd.return_value = "/tmp/scene.usd"
    mock_load_tree.return_value = kitchen_prim_tree()
    client = MagicMock()
    client.chat.completions.create.return_value = chat_response(json.dumps(response))
    resolver = ObjectReferencePrimResolver(QueryBackend(client, "test-model"))
    spec = ArenaEnvGraphSpec.model_validate(kitchen_pass1_dict())
    with pytest.raises(ValidationError, match=match):
        resolver.infer(spec, [])
