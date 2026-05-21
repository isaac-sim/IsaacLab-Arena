# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

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
