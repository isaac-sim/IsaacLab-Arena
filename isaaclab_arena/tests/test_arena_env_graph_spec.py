# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintType,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
)

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_arena_env_graph_spec_loads_pick_and_place_yaml():
    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")

    assert spec.env_name == "pick_and_place_maple_table_default"
    assert len(spec.nodes) == 6
    assert len(spec.tasks) == 1
    assert len(spec.state_specs) == 2

    table = spec.nodes_by_id["maple_table_robolab_table"]
    assert isinstance(table, ArenaEnvGraphObjectReferenceNodeSpec)
    assert table.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE
    assert table.parent == "maple_table_robolab"
    assert table.prim_path == "{ENV_REGEX_NS}/maple_table_robolab/table"
    assert table.object_type == ObjectType.RIGID

    mug = spec.nodes_by_id["mug_ycb_robolab"]
    assert mug.type == ArenaEnvGraphNodeType.OBJECT

    task = spec.tasks_by_id["pick_and_place_0"]
    assert task.initial_state_spec_id == "state_spec_0"
    assert task.success_state_spec_id == "state_spec_1"
    assert task.task_args["object"] == "rubiks_cube_hot3d_robolab"
    assert task.task_args["destination"] == "bowl_ycb_robolab"

    initial_state = spec.state_specs_by_id["state_spec_0"]
    assert isinstance(initial_state, ArenaEnvGraphStateSpec)
    assert len(initial_state.spatial_constraints) == 6
    assert len(initial_state.task_constraints) == 1

    cube_limits = initial_state.spatial_constraints[2]
    assert cube_limits.type == ArenaEnvGraphSpatialConstraintType.POSITION_LIMITS
    assert cube_limits.parent == "rubiks_cube_hot3d_robolab"
    assert cube_limits.params == {"x_min": 0.55, "x_max": 0.70, "y_min": -0.40, "y_max": -0.10}

    initial_mug_pose = initial_state.spatial_constraints[5]
    assert initial_mug_pose.type == ArenaEnvGraphSpatialConstraintType.AT_POSE
    assert initial_mug_pose.parent == "mug_ycb_robolab"
    assert initial_mug_pose.child is None
    assert initial_mug_pose.params["position_xyz"] == (0.65, 0.25, 0.85)
    assert initial_mug_pose.params["rotation_xyzw"] == (0.0, 0.0, 0.0, 1.0)

    final_state = spec.state_specs_by_id["state_spec_1"]
    in_constraint = final_state.spatial_constraints[3]
    assert in_constraint.type == ArenaEnvGraphSpatialConstraintType.IN
    assert in_constraint.parent == "bowl_ycb_robolab"
    assert in_constraint.child == "rubiks_cube_hot3d_robolab"

    final_mug_pose = final_state.spatial_constraints[4]
    assert final_mug_pose.type == ArenaEnvGraphSpatialConstraintType.AT_POSE
    assert final_mug_pose.parent == "mug_ycb_robolab"
    assert final_mug_pose.params["position_xyz"] == (0.65, 0.25, 0.85)
    assert final_mug_pose.params["rotation_xyzw"] == (0.0, 0.0, 0.0, 1.0)


def test_arena_env_graph_spec_parses_optional_task_constraints_and_at_pose():
    data = _minimal_env_graph_data()
    data["state_specs"][0]["spatial_constraints"] = [_at_pose_constraint()]
    del data["state_specs"][0]["task_constraints"]

    spec = ArenaEnvGraphSpec.from_dict(data)
    state_spec = spec.state_specs_by_id["state_0"]
    fixed_pose = state_spec.spatial_constraints[0]

    assert state_spec.task_constraints == []
    assert fixed_pose.type == ArenaEnvGraphSpatialConstraintType.AT_POSE
    assert fixed_pose.parent == "cube"
    assert fixed_pose.params["position_xyz"] == (0.1, 0.2, 0.3)
    assert fixed_pose.params["rotation_xyzw"] == (0.0, 0.0, 0.0, 1.0)


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
            "Missing required string field 'success_state_spec_id'",
        ),
        (
            "old task state map",
            lambda data: data["tasks"][0].__setitem__("state_specs", {"initial": "state_0", "final": "state_0"}),
            "must use initial_state_spec_id and success_state_spec_id",
        ),
        (
            "old task state keys",
            _add_old_task_state_keys,
            "must use initial_state_spec_id and success_state_spec_id",
        ),
        (
            "missing constraint node reference",
            lambda data: data["state_specs"][0]["task_constraints"][0].__setitem__("child", "missing_cube"),
            "unknown child node 'missing_cube'",
        ),
        (
            "missing spatial parent",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].pop("parent"),
            "Missing required string field 'parent'",
        ),
        (
            "old state edges wrapper",
            _move_state_constraints_under_edges,
            "must define spatial_constraints and task_constraints directly",
        ),
        (
            "missing node parent reference",
            lambda data: data["nodes"][1].__setitem__("parent", "missing_background"),
            "unknown parent 'missing_background'",
        ),
        (
            "unknown object type",
            lambda data: data["nodes"][1].__setitem__("object_type", "unknown"),
            "Unknown object_type 'unknown'",
        ),
        (
            "object_reference missing parent",
            lambda data: data["nodes"][1].pop("parent"),
            "Missing required string field 'parent'",
        ),
        (
            "object_reference missing prim_path",
            lambda data: data["nodes"][1].pop("prim_path"),
            "Missing required string field 'prim_path'",
        ),
        (
            "object_reference missing object_type",
            lambda data: data["nodes"][1].pop("object_type"),
            "Missing required field 'object_type'",
        ),
        (
            "unknown node type",
            lambda data: data["nodes"][0].__setitem__("type", "unknown"),
            "Unknown type 'unknown'",
        ),
        (
            "unknown spatial constraint type",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].__setitem__("type", "unknown"),
            "Unknown type 'unknown'",
        ),
        (
            "unknown task constraint type",
            lambda data: data["state_specs"][0]["task_constraints"][0].__setitem__("type", "unknown"),
            "Unknown type 'unknown'",
        ),
        (
            "invalid at_pose position",
            lambda data: data["state_specs"][0]["spatial_constraints"].append(
                _at_pose_constraint(position_xyz=[0.1, 0.2])
            ),
            "Field 'position_xyz' must contain 3 numbers",
        ),
    ]

    for label, mutate, error_match in cases:
        data = _minimal_env_graph_data()
        mutate(data)

        try:
            ArenaEnvGraphSpec.from_dict(data)
        except AssertionError as exc:
            assert error_match in str(exc), label
        else:
            raise AssertionError(f"{label}: expected AssertionError")


def _minimal_env_graph_data():
    return {
        "env_name": "minimal_env_graph",
        "nodes": [
            {"id": "robot", "name": "robot", "type": "embodiment"},
            # Kept at index 1 so the bad-data mutation lambdas below can address it.
            {
                "id": "table",
                "name": "table",
                "type": "object_reference",
                "parent": "background",
                "prim_path": "{ENV_REGEX_NS}/background/table",
                "object_type": "rigid",
            },
            {"id": "background", "name": "background", "type": "background"},
            {"id": "cube", "name": "cube", "type": "object"},
        ],
        "tasks": [{
            "id": "task_0",
            "type": "pick_and_place",
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


def _at_pose_constraint(position_xyz=None, rotation_xyzw=None):
    return {
        "id": "cube_fixed_pose",
        "type": "at_pose",
        "parent": "cube",
        "params": {
            "position_xyz": [0.1, 0.2, 0.3] if position_xyz is None else position_xyz,
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0] if rotation_xyzw is None else rotation_xyzw,
        },
    }


def _add_old_task_state_keys(data):
    data["tasks"][0]["initial_state_spec"] = "state_0"
    data["tasks"][0]["success_state_spec"] = "state_0"


def _move_state_constraints_under_edges(data):
    state_spec = data["state_specs"][0]
    state_spec["edges"] = {
        "spatial_constraints": state_spec.pop("spatial_constraints"),
        "task_constraints": state_spec.pop("task_constraints"),
    }
