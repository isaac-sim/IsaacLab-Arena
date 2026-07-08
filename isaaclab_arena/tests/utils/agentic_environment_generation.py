# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for agentic environment generation tests."""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    AssetCatalogue,
    RelationCatalogue,
    TaskCatalogue,
)
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord

_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


def load_test_yaml(name: str) -> dict[str, Any]:
    """Load a YAML fixture from ``isaaclab_arena/tests/test_data``."""
    path = _TEST_DATA_DIR / name
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"Expected dict in {path}, got {type(data).__name__}"
    return data


def load_test_json(name: str) -> dict[str, Any]:
    """Load a JSON fixture from ``isaaclab_arena/tests/test_data``."""
    path = _TEST_DATA_DIR / name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), f"Expected dict in {path}, got {type(data).__name__}"
    return data


def minimal_spec_dict() -> dict[str, Any]:
    """Return the minimal maple-table pass-1 graph spec used in agent tests."""
    return load_test_yaml("minimal_maple_table_env_graph.yaml")


def kitchen_pass1_dict() -> dict[str, Any]:
    """Return the kitchen pass-1 graph spec with unresolved object references."""
    return load_test_yaml("kitchen_pass1_env_graph.yaml")


def kitchen_resolve_response() -> dict[str, Any]:
    """Return the pass-2 resolver LLM response for the kitchen object references."""
    return load_test_json("kitchen_resolve_object_references.json")


def kitchen_prim_tree() -> list[UsdPrimRecord]:
    """Return the mocked kitchen USD prim tree for pass-2 resolver tests."""
    return [
        UsdPrimRecord("counter_right_main_group/top_geometry", "base"),
        UsdPrimRecord("fridge_main_group", "articulation", ("fridge_door_joint",)),
    ]


def chat_response(
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
):
    """Build a nested mock matching the OpenAI chat-completion response shape."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].finish_reason = finish_reason
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning_content
    return resp


def catalog(text: str) -> AssetCatalogue:
    """Return an asset catalogue that renders ``text`` in the user message."""
    catalogue = AssetCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def relation_catalog(text: str) -> RelationCatalogue:
    """Return a relation catalogue that renders ``text`` in the user message."""
    catalogue = RelationCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def task_catalog(text: str) -> TaskCatalogue:
    """Return a task catalogue that renders ``text`` in the user message."""
    catalogue = TaskCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue
