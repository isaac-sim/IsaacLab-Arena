# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from pydantic import ValidationError

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
)
from isaaclab_arena.environments.graph_spec_utils import relation_class_for_spatial_constraint_type
from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On, PositionLimits

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_arena_env_graph_spec_loads_pick_and_place_yaml():
    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")

    assert spec.env_name == "pick_and_place_maple_table_default"
    assert len(spec.nodes) == 6
    assert len(spec.tasks) == 2
    assert len(spec.state_specs) == 3

    table = spec.nodes_by_id["maple_table_robolab_table"]
    assert isinstance(table, ArenaEnvGraphObjectReferenceNodeSpec)
    assert table.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE
    assert table.parent == "maple_table_robolab"
    assert table.prim_path == "{ENV_REGEX_NS}/maple_table_robolab/table"
    assert table.object_type == ObjectType.RIGID

    mug = spec.nodes_by_id["mug_ycb_robolab"]
    assert mug.type == ArenaEnvGraphNodeType.OBJECT

    task = spec.tasks_by_id["pick_and_place_0"]
    assert task.type == "PickAndPlaceTask"
    assert TaskRegistry().is_registered(task.type)
    assert task.initial_state_spec_id == "state_spec_0"
    assert task.success_state_spec_id == "state_spec_1"
    assert task.task_args["pick_up_object"] == "rubiks_cube_hot3d_robolab"
    assert task.task_args["destination_location"] == "bowl_ycb_robolab"

    second_task = spec.tasks_by_id["pick_and_place_1"]
    assert second_task.type == "PickAndPlaceTask"
    assert second_task.initial_state_spec_id == "state_spec_1"
    assert second_task.success_state_spec_id == "state_spec_2"
    assert second_task.task_args["pick_up_object"] == "mug_ycb_robolab"

    initial_state = spec.state_specs_by_id["state_spec_0"]
    assert isinstance(initial_state, ArenaEnvGraphStateSpec)
    assert len(initial_state.spatial_constraints) == 6
    assert len(initial_state.task_constraints) == 1

    cube_limits = initial_state.spatial_constraints[2]
    assert cube_limits.type == "position_limits"
    assert cube_limits.parent == "rubiks_cube_hot3d_robolab"
    assert cube_limits.params == {"x_min": 0.55, "x_max": 0.70, "y_min": -0.40, "y_max": -0.10}

    initial_mug_position = initial_state.spatial_constraints[5]
    assert initial_mug_position.type == "at_position"
    assert initial_mug_position.parent == "mug_ycb_robolab"
    assert initial_mug_position.child is None
    assert initial_mug_position.params == {"x": 0.65, "y": 0.25, "z": 0.85}

    final_state = spec.state_specs_by_id["state_spec_1"]
    cube_on_bowl = final_state.spatial_constraints[3]
    assert cube_on_bowl.type == "on"
    assert cube_on_bowl.parent == "bowl_ycb_robolab"
    assert cube_on_bowl.child == "rubiks_cube_hot3d_robolab"

    final_mug_position = final_state.spatial_constraints[4]
    assert final_mug_position.type == "at_position"
    assert final_mug_position.parent == "mug_ycb_robolab"
    assert final_mug_position.params == {"x": 0.65, "y": 0.25, "z": 0.85}

    table_anchor = initial_state.spatial_constraints[0]
    assert table_anchor.type == "is_anchor"
    assert relation_class_for_spatial_constraint_type(table_anchor.type) is IsAnchor
    assert relation_class_for_spatial_constraint_type(cube_limits.type) is PositionLimits
    assert relation_class_for_spatial_constraint_type(initial_mug_position.type) is AtPosition
    assert relation_class_for_spatial_constraint_type(cube_on_bowl.type) is On


def test_arena_env_graph_spec_parses_optional_task_constraints_and_at_position():
    data = _minimal_env_graph_data()
    data["state_specs"][0]["spatial_constraints"] = [_at_position_constraint()]
    del data["state_specs"][0]["task_constraints"]

    spec = ArenaEnvGraphSpec.from_dict(data)
    assert spec.tasks_by_id["task_0"].type == "PickAndPlaceTask"
    state_spec = spec.state_specs_by_id["state_0"]
    fixed_position = state_spec.spatial_constraints[0]

    assert state_spec.task_constraints == []
    assert fixed_position.type == "at_position"
    assert fixed_position.parent == "cube"
    assert fixed_position.params == {"x": 0.1, "y": 0.2, "z": 0.3}


def test_arena_env_graph_spec_validate_rejects_mutated_missing_reference():
    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    spec.state_specs[0].spatial_constraints[0].parent = "missing_table"

    with pytest.raises(AssertionError, match="unknown parent node 'missing_table'"):
        spec.validate()


def test_arena_env_graph_spec_validate_rejects_mutated_invalid_relationship_shape():
    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    constraint = spec.state_specs[0].spatial_constraints[0]
    constraint.type = "on"

    with pytest.raises(AssertionError, match="requires a child node"):
        spec.validate()


def test_from_yaml_rejects_missing_path_with_clear_message():
    with pytest.raises(ValueError, match="Env graph spec YAML not found"):
        ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "does_not_exist.yaml")


def test_arena_env_graph_spec_rejects_invalid_data():
    cases = [
        (
            "duplicate node id",
            lambda data: data["nodes"].append({"id": "table", "name": "duplicate_table", "type": "background"}),
            "Duplicate env graph ids",
        ),
        (
            "duplicate id across spec types",
            lambda data: data["tasks"][0].__setitem__("id", "table"),
            "Duplicate env graph ids",
        ),
        (
            "duplicate constraint id",
            lambda data: data["state_specs"][0]["task_constraints"][0].__setitem__("id", "table_is_anchor"),
            "Duplicate env graph ids",
        ),
        (
            "missing task state reference",
            lambda data: data["tasks"][0].__setitem__("initial_state_spec_id", "missing_state"),
            "unknown state spec 'missing_state'",
        ),
        (
            "missing required task state spec id",
            lambda data: data["tasks"][0].pop("success_state_spec_id"),
            "success_state_spec_id",
        ),
        (
            "missing constraint node reference",
            lambda data: data["state_specs"][0]["task_constraints"][0].__setitem__("child", "missing_cube"),
            "unknown child node 'missing_cube'",
        ),
        (
            "missing spatial parent",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].pop("parent"),
            "parent",
        ),
        (
            "relationship missing child",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].__setitem__("type", "on"),
            "requires a child node",
        ),
        (
            "unary relationship with child",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].__setitem__("child", "cube"),
            "must not define a child node",
        ),
        (
            "missing node parent reference",
            lambda data: data["nodes"][2].__setitem__("parent", "missing_background"),
            "unknown parent 'missing_background'",
        ),
        (
            "unknown object type",
            lambda data: data["nodes"][2].__setitem__("object_type", "unknown"),
            "object_type",
        ),
        (
            "object_reference missing parent",
            lambda data: data["nodes"][2].pop("parent"),
            "parent",
        ),
        (
            "object_reference missing prim_path",
            lambda data: data["nodes"][2].pop("prim_path"),
            "prim_path",
        ),
        (
            "object_reference missing object_type",
            lambda data: data["nodes"][2].pop("object_type"),
            "object_type",
        ),
        (
            "unknown node type",
            lambda data: data["nodes"][0].__setitem__("type", "unknown"),
            "type",
        ),
        (
            "unknown task type",
            lambda data: data["tasks"][0].__setitem__("type", "UnknownTask"),
            "Unknown task type 'UnknownTask'",
        ),
        (
            "unknown spatial constraint type",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].__setitem__("type", "unknown"),
            "Unknown spatial constraint type 'unknown'",
        ),
        (
            "unknown task constraint type",
            lambda data: data["state_specs"][0]["task_constraints"][0].__setitem__("type", "unknown"),
            "type",
        ),
    ]

    for label, mutate, error_match in cases:
        data = _minimal_env_graph_data()
        mutate(data)

        with pytest.raises(ValidationError) as exc_info:
            ArenaEnvGraphSpec.from_dict(data)
        assert error_match in str(exc_info.value), label


def _minimal_env_graph_data():
    return {
        "env_name": "minimal_env_graph",
        "nodes": [
            {"id": "robot", "name": "robot", "type": "embodiment"},
            {"id": "background", "name": "background", "type": "background"},
            {
                "id": "table",
                "name": "table",
                "type": "object_reference",
                "parent": "background",
                "prim_path": "{ENV_REGEX_NS}/background/table",
                "object_type": "rigid",
            },
            {"id": "cube", "name": "cube", "type": "object"},
        ],
        "tasks": [{
            "id": "task_0",
            "type": "PickAndPlaceTask",
            "initial_state_spec_id": "state_0",
            "success_state_spec_id": "state_0",
        }],
        "state_specs": [{
            "id": "state_0",
            "spatial_constraints": [{"id": "table_is_anchor", "type": "is_anchor", "parent": "table"}],
            "task_constraints": [{
                "id": "robot_reach_cube",
                "type": "reach",
                "parent": "robot",
                "child": "cube",
            }],
        }],
    }


def _at_position_constraint(x=0.1, y=0.2, z=0.3):
    return {
        "id": "cube_fixed_position",
        "type": "at_position",
        "parent": "cube",
        "params": {"x": x, "y": y, "z": z},
    }
