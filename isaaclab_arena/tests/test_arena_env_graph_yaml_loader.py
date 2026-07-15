# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for env-graph YAML include loading."""

from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import (
    deep_merge_env_graph_dicts,
    load_env_graph_spec_dict,
)

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_deep_merge_nested_dicts_and_replaces_lists():
    base = {"embodiment": {"id": "droid", "params": {}}, "objects": [{"id": "a"}]}
    override = {"embodiment": {"registry_name": "droid_abs_joint_pos"}, "tasks": [{"kind": "NoTask"}]}
    merged = deep_merge_env_graph_dicts(base, override)
    assert merged["embodiment"] == {"id": "droid", "params": {}, "registry_name": "droid_abs_joint_pos"}
    assert merged["objects"] == [{"id": "a"}]
    assert merged["tasks"] == [{"kind": "NoTask"}]


def test_load_env_graph_spec_dict_resolves_yaml_includes():
    data = load_env_graph_spec_dict(TEST_DATA_DIR / "robolab_task_overlay.yaml")
    assert data["env_name"] == "banana_in_bowl"
    assert len(data["objects"]) == 2
    assert len(data["task"]["subtasks"]) == 1
    assert data["background"]["registry_name"] == "maple_table_robolab"
    assert "external_yamls" not in data


def test_arena_env_graph_spec_from_yaml_resolves_robolab_task_include():
    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "robolab_task_overlay.yaml")
    assert spec.env_name == "banana_in_bowl"
    assert len(spec.objects) == 2
    assert len(spec.task.subtasks) == 1
    assert spec.task.subtasks[0].params["pick_up_object"] == "banana"


def test_load_env_graph_spec_dict_rejects_missing_include(tmp_path):
    path = tmp_path / "_missing_include.yaml"
    path.write_text("external_yamls:\n  - does_not_exist.yaml\nenv_name: broken\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Env graph spec YAML not found"):
        load_env_graph_spec_dict(path)


def test_robolab_task_yaml_loads_scene_include():
    task_path = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
    spec = ArenaEnvGraphSpec.from_yaml(task_path)
    assert spec.env_name == "banana_in_bowl"
    assert len(spec.objects) == 2
    assert spec.task.subtasks[0].params["pick_up_object"] == "banana"


def test_load_env_graph_spec_dict_rejects_cyclic_includes(tmp_path):
    a = tmp_path / "_cycle_a.yaml"
    b = tmp_path / "_cycle_b.yaml"
    a.write_text(f"external_yamls:\n  - {b}\nenv_name: a\n", encoding="utf-8")
    b.write_text(f"external_yamls:\n  - {a}\nenv_name: b\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Cyclic env graph spec external_yamls include"):
        load_env_graph_spec_dict(a)
