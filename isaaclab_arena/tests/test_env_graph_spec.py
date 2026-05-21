# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path

import pytest

from isaaclab_arena.environments.env_graph_spec import EnvGraphSpec, EnvGraphStateSpec

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_env_graph_spec_loads_pick_and_place_yaml():
    spec = EnvGraphSpec.from_yaml(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")

    assert spec.name == "pick_and_place_maple_table_default"
    assert len(spec.nodes) == 5
    assert len(spec.tasks) == 1
    assert len(spec.state_specs) == 2

    table = spec.nodes_by_id["maple_table_robolab_table"]
    assert table.type == "object_reference"
    assert table.parent == "maple_table_robolab"
    assert table.prim_path == "{ENV_REGEX_NS}/maple_table_robolab/table"
    assert table.object_type == "rigid"

    task = spec.tasks_by_id["pick_and_place_0"]
    assert task.state_specs == {"initial": "state_spec_0", "final": "state_spec_1"}
    assert task.task_args["object"] == "rubiks_cube_hot3d_robolab"
    assert task.task_args["destination"] == "bowl_ycb_robolab"
    assert task.task_args["episode_length_s"] == 20.0

    initial_state = spec.state_specs_by_id["state_spec_0"]
    assert isinstance(initial_state, EnvGraphStateSpec)
    assert len(initial_state.edges.spatial_constraints) == 5
    assert len(initial_state.edges.task_constraints) == 1

    cube_limits = initial_state.edges.spatial_constraints[2]
    assert cube_limits.type == "position_limits"
    assert cube_limits.child == "rubiks_cube_hot3d_robolab"
    assert cube_limits.params == {
        "x_min": 0.55,
        "x_max": 0.70,
        "y_min": -0.40,
        "y_max": -0.10,
    }

    final_state = spec.state_specs_by_id["state_spec_1"]
    assert isinstance(final_state, EnvGraphStateSpec)
    in_constraint = final_state.edges.spatial_constraints[3]
    assert in_constraint.type == "in"
    assert in_constraint.parent == "bowl_ycb_robolab"
    assert in_constraint.child == "rubiks_cube_hot3d_robolab"

    reach_constraint = final_state.edges.task_constraints[0]
    assert reach_constraint.type == "reach"
    assert reach_constraint.parent == "droid_abs_joint_pos"
    assert reach_constraint.child == "bowl_ycb_robolab"


def test_env_graph_spec_rejects_duplicate_ids():
    data = _minimal_env_graph_data()
    data["nodes"].append({"id": "table", "name": "duplicate_table", "type": "object_reference"})

    with pytest.raises(AssertionError, match="Duplicate node ids"):
        EnvGraphSpec.from_dict(data)


def test_env_graph_spec_rejects_missing_task_state_reference():
    data = _minimal_env_graph_data()
    data["tasks"][0]["state_specs"]["initial"] = "missing_state"

    with pytest.raises(AssertionError, match="unknown state spec 'missing_state'"):
        EnvGraphSpec.from_dict(data)


def test_env_graph_spec_rejects_missing_constraint_node_reference():
    data = _minimal_env_graph_data()
    data["state_specs"][0]["edges"]["task_constraints"][0]["child"] = "missing_cube"

    with pytest.raises(AssertionError, match="unknown child node 'missing_cube'"):
        EnvGraphSpec.from_dict(data)


def _minimal_env_graph_data():
    return deepcopy(
        {
            "name": "minimal_env_graph",
            "nodes": [
                {"id": "robot", "name": "robot", "type": "embodiment"},
                {"id": "table", "name": "table", "type": "object_reference"},
                {"id": "cube", "name": "cube", "type": "rigid_object"},
            ],
            "tasks": [
                {
                    "id": "task_0",
                    "name": "task_0",
                    "type": "pick_and_place",
                    "state_specs": {"initial": "state_0"},
                }
            ],
            "state_specs": [
                {
                    "id": "state_0",
                    "name": "state_0",
                    "edges": {
                        "spatial_constraints": [
                            {"id": "table_is_anchor", "type": "is_anchor", "parent": "table"}
                        ],
                        "task_constraints": [
                            {
                                "id": "robot_reach_cube",
                                "type": "reach",
                                "parent": "robot",
                                "child": "cube",
                            }
                        ],
                    },
                }
            ],
        }
    )
