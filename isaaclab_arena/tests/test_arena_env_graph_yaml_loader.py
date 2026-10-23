# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for env-graph YAML include loading."""

import yaml
from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import _merge_env_graph_dicts, load_env_graph_spec_dict

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_merge_combines_disjoint_keys():
    base = {"embodiment": {"id": "droid", "params": {}}, "objects": [{"id": "a"}]}
    override = {"background": {"id": "table"}, "tasks": [{"kind": "NoTask"}]}
    merged = _merge_env_graph_dicts([("base.yaml", base), ("override.yaml", override)])
    assert merged == {
        "embodiment": {"id": "droid", "params": {}},
        "objects": [{"id": "a"}],
        "background": {"id": "table"},
        "tasks": [{"kind": "NoTask"}],
    }


def test_merge_does_not_modify_inputs():
    base = {"objects": [{"id": "a"}], "env_cfg_override": {"sim": {"dt": 0.01}}}
    override = {"objects": [{"id": "b"}], "env_cfg_override": {"sim": {"render_interval": 2}}}
    _merge_env_graph_dicts([("base.yaml", base), ("override.yaml", override)])
    assert base == {"objects": [{"id": "a"}], "env_cfg_override": {"sim": {"dt": 0.01}}}
    assert override == {"objects": [{"id": "b"}], "env_cfg_override": {"sim": {"render_interval": 2}}}


def test_merge_combines_different_leaves_under_the_same_dict():
    first = {"env_cfg_override": {"sim": {"physics": {"max_position_iteration_count": 32}}}}
    second = {"env_cfg_override": {"sim": {"physics": {"max_velocity_iteration_count": 1}, "dt": 0.01}}}
    merged = _merge_env_graph_dicts([("first.yaml", first), ("second.yaml", second)])
    assert merged == {
        "env_cfg_override": {
            "sim": {"physics": {"max_position_iteration_count": 32, "max_velocity_iteration_count": 1}, "dt": 0.01}
        }
    }


def test_merge_concatenates_id_lists_in_source_order():
    sources = [
        ("a.yaml", {"objects": [{"id": "banana"}, {"id": "bowl"}]}),
        ("b.yaml", {"objects": [{"id": "light"}]}),
        ("c.yaml", {"objects": [{"id": "sphere_light"}]}),
    ]
    merged = _merge_env_graph_dicts(sources)
    assert [obj["id"] for obj in merged["objects"]] == ["banana", "bowl", "light", "sphere_light"]


def test_merge_rejects_duplicate_top_level_leaf():
    sources = [("a.yaml", {"env_name": "one"}), ("b.yaml", {"env_name": "two"})]
    with pytest.raises(ValueError) as exc_info:
        _merge_env_graph_dicts(sources)
    message = str(exc_info.value)
    assert "'env_name'" in message
    assert "a.yaml" in message and "b.yaml" in message
    assert "'one'" in message and "'two'" in message


def test_merge_rejects_nested_leaf_defined_twice_and_reports_full_path():
    first = {"env_cfg_override": {"sim": {"physics": {"max_position_iteration_count": 32}}}}
    second = {"env_cfg_override": {"sim": {"physics": {"max_position_iteration_count": 16}}}}
    with pytest.raises(ValueError) as exc_info:
        _merge_env_graph_dicts([("defaults.yaml", first), ("task.yaml", second)])
    message = str(exc_info.value)
    assert "'env_cfg_override.sim.physics.max_position_iteration_count'" in message
    assert "defaults.yaml (value 32)" in message
    assert "task.yaml (value 16)" in message


def test_merge_rejects_identical_values():
    sources = [("a.yaml", {"embodiment": {"id": "droid"}}), ("b.yaml", {"embodiment": {"id": "droid"}})]
    with pytest.raises(ValueError, match="'embodiment.id'"):
        _merge_env_graph_dicts(sources)


def test_merge_reports_the_source_that_first_defined_a_conflicting_value():
    sources = [
        ("a.yaml", {"x": {"y": 1}}),
        ("b.yaml", {"x": {"z": 2}}),
        ("c.yaml", {"x": {"z": 3}}),
    ]
    with pytest.raises(ValueError, match=r"'x\.z': defined in both b.yaml .* and c.yaml"):
        _merge_env_graph_dicts(sources)


def test_merge_rejects_duplicate_id_in_id_list():
    sources = [
        ("scene.yaml", {"objects": [{"id": "banana"}, {"id": "light"}]}),
        ("defaults.yaml", {"objects": [{"id": "light"}]}),
    ]
    with pytest.raises(ValueError) as exc_info:
        _merge_env_graph_dicts(sources)
    message = str(exc_info.value)
    assert "'objects[id=light]'" in message
    assert "scene.yaml" in message and "defaults.yaml" in message


def test_merge_reports_the_source_of_an_id_added_by_an_earlier_source():
    sources = [
        ("a.yaml", {"objects": [{"id": "banana"}]}),
        ("b.yaml", {"objects": [{"id": "light"}]}),
        ("c.yaml", {"objects": [{"id": "light"}]}),
    ]
    with pytest.raises(ValueError) as exc_info:
        _merge_env_graph_dicts(sources)
    message = str(exc_info.value)
    assert "'objects[id=light]'" in message
    assert "b.yaml" in message and "c.yaml" in message


def test_merge_gives_yaml_aliases_independent_copies():
    included = yaml.safe_load("env_cfg_override: {a: &shared {x: 1}, b: *shared}")
    task = {"env_cfg_override": {"a": {"y": 2}}}
    merged = _merge_env_graph_dicts([("defaults.yaml", included), ("task.yaml", task)])
    assert merged["env_cfg_override"] == {"a": {"x": 1, "y": 2}, "b": {"x": 1}}


def test_merge_rejects_lists_without_ids():
    sources = [("a.yaml", {"hdr_names": ["x"]}), ("b.yaml", {"hdr_names": ["y"]})]
    with pytest.raises(ValueError, match="'hdr_names'"):
        _merge_env_graph_dicts(sources)


def test_merge_rejects_mixed_types_at_the_same_path():
    sources = [("a.yaml", {"background": {"id": "table"}}), ("b.yaml", {"background": "table"})]
    with pytest.raises(ValueError, match="'background'"):
        _merge_env_graph_dicts(sources)


def test_merge_rejects_scalar_against_id_list():
    sources = [("a.yaml", {"objects": [{"id": "a"}]}), ("b.yaml", {"objects": None})]
    with pytest.raises(ValueError, match="'objects'"):
        _merge_env_graph_dicts(sources)


def test_load_env_graph_spec_dict_resolves_yaml_includes():
    data = load_env_graph_spec_dict(TEST_DATA_DIR / "robolab_task_overlay.yaml")
    assert data["env_name"] == "banana_in_bowl"
    assert len(data["objects"]) == 2
    assert len(data["task"]["subtasks"]) == 1
    assert data["background"]["registry_name"] == "maple_table_robolab"
    assert "external_yaml" not in data


def test_arena_env_graph_spec_from_yaml_resolves_robolab_task_include():
    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "robolab_task_overlay.yaml")
    assert spec.env_name == "banana_in_bowl"
    assert len(spec.objects) == 2
    assert len(spec.task.subtasks) == 1
    assert spec.task.subtasks[0].params["pick_up_object"] == "banana"


def test_load_env_graph_spec_dict_rejects_missing_include(tmp_path):
    path = tmp_path / "_missing_include.yaml"
    path.write_text("external_yaml: does_not_exist.yaml\nenv_name: broken\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Env graph spec YAML not found"):
        load_env_graph_spec_dict(path)


def test_robolab_task_yaml_loads_scene_include():
    task_path = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
    spec = ArenaEnvGraphSpec.from_yaml(task_path)
    assert spec.env_name == "banana_in_bowl"
    assert len(spec.objects) == 2
    assert spec.task.subtasks[0].params["pick_up_object"] == "banana"


def test_load_env_graph_spec_dict_rejects_nested_includes(tmp_path):
    entry = tmp_path / "_entry.yaml"
    mid = tmp_path / "_mid.yaml"
    leaf = tmp_path / "_leaf.yaml"
    leaf.write_text("env_name: leaf\n", encoding="utf-8")
    mid.write_text("external_yaml: _leaf.yaml\nembodiment: {}\n", encoding="utf-8")
    entry.write_text("external_yaml: _mid.yaml\nenv_name: entry\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Nested 'external_yaml' is not allowed"):
        load_env_graph_spec_dict(entry)


def test_load_env_graph_spec_dict_merges_a_list_of_includes(tmp_path):
    (tmp_path / "scene.yaml").write_text("objects:\n- id: banana\nbackground:\n  id: table\n", encoding="utf-8")
    (tmp_path / "defaults.yaml").write_text(
        "objects:\n- id: light\nenv_cfg_override:\n  sim:\n    dt: 0.01\n", encoding="utf-8"
    )
    entry = tmp_path / "task.yaml"
    entry.write_text(
        "external_yaml:\n- scene.yaml\n- defaults.yaml\nenv_name: demo\nobjects:\n- id: bowl\n",
        encoding="utf-8",
    )
    data = load_env_graph_spec_dict(entry)
    assert [obj["id"] for obj in data["objects"]] == ["banana", "light", "bowl"]
    assert data["env_cfg_override"] == {"sim": {"dt": 0.01}}
    assert data["env_name"] == "demo"
    assert "external_yaml" not in data


def test_load_env_graph_spec_dict_error_names_the_conflicting_files(tmp_path):
    (tmp_path / "scene.yaml").write_text("env_cfg_override:\n  sim:\n    dt: 0.01\n", encoding="utf-8")
    entry = tmp_path / "task.yaml"
    entry.write_text("external_yaml: scene.yaml\nenv_cfg_override:\n  sim:\n    dt: 0.02\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_env_graph_spec_dict(entry)
    message = str(exc_info.value)
    assert "'env_cfg_override.sim.dt'" in message
    assert str(tmp_path / "scene.yaml") in message
    assert str(entry) in message


def test_load_env_graph_spec_dict_rejects_nested_include_in_a_list(tmp_path):
    (tmp_path / "ok.yaml").write_text("env_name: ok\n", encoding="utf-8")
    (tmp_path / "nested.yaml").write_text("external_yaml: ok.yaml\n", encoding="utf-8")
    entry = tmp_path / "entry.yaml"
    entry.write_text("external_yaml: [ok.yaml, nested.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Nested 'external_yaml' is not allowed"):
        load_env_graph_spec_dict(entry)


@pytest.mark.parametrize("include", ["[]", "3", "[a.yaml, 3]"])
def test_load_env_graph_spec_dict_rejects_malformed_include(tmp_path, include):
    entry = tmp_path / "entry.yaml"
    entry.write_text(f"external_yaml: {include}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external_yaml"):
        load_env_graph_spec_dict(entry)
