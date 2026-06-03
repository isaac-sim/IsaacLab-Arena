# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintType,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
)
from isaaclab_arena.environments.graph_spec_utils import relation_class_for_spatial_constraint_type
from isaaclab_arena.relations.relations import IsAnchor, PositionLimits

TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Spatial-constraint enum members that intentionally have no registered relation:
# AT_POSE is applied via set_initial_pose(), and IN is not yet supported by the solver.
# TODO(xinjieyao, 2026-05-28): drop these once AT_POSE and IN gain relation classes.
_RELATIONLESS_CONSTRAINT_TYPES = {
    ArenaEnvGraphSpatialConstraintType.AT_POSE,
    ArenaEnvGraphSpatialConstraintType.IN,
}


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
    assert task.initial_state_spec_id == "state_spec_0"
    assert task.success_state_spec_id == "state_spec_1"
    assert task.task_args["pick_up_object"] == "rubiks_cube_hot3d_robolab"
    assert task.task_args["destination_location"] == "bowl_ycb_robolab"

    second_task = spec.tasks_by_id["pick_and_place_1"]
    assert second_task.initial_state_spec_id == "state_spec_1"
    assert second_task.success_state_spec_id == "state_spec_2"
    assert second_task.task_args["pick_up_object"] == "mug_ycb_robolab"

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

    table_anchor = initial_state.spatial_constraints[0]
    assert table_anchor.type == ArenaEnvGraphSpatialConstraintType.IS_ANCHOR
    assert relation_class_for_spatial_constraint_type(table_anchor.type) is IsAnchor
    assert relation_class_for_spatial_constraint_type(cube_limits.type) is PositionLimits
    assert (
        relation_class_for_spatial_constraint_type(initial_mug_pose.type) is None
    )  # at_pose: handled via set_initial_pose
    assert relation_class_for_spatial_constraint_type(in_constraint.type) is None  # in: not yet supported by solver


def test_registered_relations_match_spatial_constraint_enum():
    """Registered relations and the spatial-constraint enum must stay in one-to-one sync.

    Each registered RelationBase subclass is keyed by its `name`, which must equal the
    `value` of a ArenaEnvGraphSpatialConstraintType member (so spec lookups resolve), and
    every solver-backed enum member must have a relation. AT_POSE and IN are excluded —
    see _RELATIONLESS_CONSTRAINT_TYPES. This guards against adding one side without the
    other.
    """
    # Importing the module ran the @register_object_relation decorators at file top.
    registered_names = set(ObjectRelationLibraryRegistry().get_all_keys())
    enum_values = {
        constraint.value
        for constraint in ArenaEnvGraphSpatialConstraintType
        if constraint not in _RELATIONLESS_CONSTRAINT_TYPES
    }

    assert registered_names == enum_values, (
        "Registered relations and spatial-constraint enum are out of sync.\n"
        f"  relations missing an enum member: {sorted(registered_names - enum_values)}\n"
        f"  enum members missing a relation:  {sorted(enum_values - registered_names)}\n"
        "  (AT_POSE and IN are intentionally excluded via _RELATIONLESS_CONSTRAINT_TYPES.)"
    )


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


def test_arena_env_graph_spec_validate_rejects_mutated_missing_reference():
    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    spec.state_specs[0].spatial_constraints[0].parent = "missing_table"

    with pytest.raises(AssertionError, match="unknown parent node 'missing_table'"):
        spec.validate()


def test_arena_env_graph_spec_validate_rejects_mutated_invalid_relationship_shape():
    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    constraint = spec.state_specs[0].spatial_constraints[0]
    constraint.type = ArenaEnvGraphSpatialConstraintType.ON

    with pytest.raises(AssertionError, match="requires a child node"):
        spec.validate()


def test_cli_override_specs_parsed_from_yaml():
    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")

    overrides = {override.arg: override for override in spec.cli_override_specs}
    assert overrides["embodiment"].target_node_id == "droid_abs_joint_pos"
    assert overrides["object"].target_node_id == "rubiks_cube_hot3d_robolab"


def test_add_cli_override_args_registers_declared_flags():
    import argparse

    from isaaclab_arena.environments.graph_spec_utils import add_cli_override_args

    parser = argparse.ArgumentParser()
    specs = ArenaEnvGraphSpec.read_cli_override_specs(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")
    add_cli_override_args(parser, specs)

    args = parser.parse_args(["--object", "dex_cube"])
    assert args.object == "dex_cube"
    # Unset flags default to None.
    assert args.embodiment is None


def test_apply_cli_override_args_swaps_declared_target_node_names():
    import argparse

    data = _minimal_env_graph_data()
    data["cli_override_specs"] = [
        {"arg": "object", "target_node_id": "cube"},
        {"arg": "embodiment", "target_node_id": "robot"},
    ]
    spec = ArenaEnvGraphSpec.from_dict(data)

    spec.apply_cli_override_args(argparse.Namespace(object="dex_cube", embodiment="franka_ik"))

    # The asset `name` is swapped; the `id` (and every edge that references it) is untouched.
    assert spec.nodes_by_id["cube"].name == "dex_cube"
    assert spec.nodes_by_id["robot"].name == "franka_ik"
    assert spec.state_specs[0].task_constraints[0].child == "cube"
    assert spec.state_specs[0].task_constraints[0].parent == "robot"


def test_apply_cli_override_args_leaves_unset_flags_as_authored():
    import argparse

    data = _minimal_env_graph_data()
    data["cli_override_specs"] = [{"arg": "object", "target_node_id": "cube"}]
    spec = ArenaEnvGraphSpec.from_dict(data)

    spec.apply_cli_override_args(argparse.Namespace(object=None))

    assert spec.nodes_by_id["cube"].name == "cube"


def test_validate_rejects_cli_override_targeting_unknown_node():
    data = _minimal_env_graph_data()
    data["cli_override_specs"] = [{"arg": "object", "target_node_id": "missing_node"}]

    with pytest.raises(AssertionError, match="targets unknown node 'missing_node'"):
        ArenaEnvGraphSpec.from_dict(data)


def test_validate_rejects_duplicate_cli_override_args():
    data = _minimal_env_graph_data()
    data["cli_override_specs"] = [
        {"arg": "object", "target_node_id": "cube"},
        {"arg": "object", "target_node_id": "robot"},
    ]

    with pytest.raises(AssertionError, match="Duplicate cli_override arg '--object'"):
        ArenaEnvGraphSpec.from_dict(data)


def test_from_yaml_rejects_missing_path_with_clear_message():
    with pytest.raises(AssertionError, match="Env graph spec YAML not found"):
        ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "does_not_exist.yaml")


def test_read_cli_override_specs_rejects_missing_path_with_clear_message():
    with pytest.raises(AssertionError, match="Env graph spec YAML not found"):
        ArenaEnvGraphSpec.read_cli_override_specs(TEST_DATA_DIR / "does_not_exist.yaml")


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
            "old state edges wrapper",
            _move_state_constraints_under_edges,
            "must define spatial_constraints and task_constraints directly",
        ),
        (
            "missing node parent reference",
            lambda data: data["nodes"][2].__setitem__("parent", "missing_background"),
            "unknown parent 'missing_background'",
        ),
        (
            "unknown object type",
            lambda data: data["nodes"][2].__setitem__("object_type", "unknown"),
            "Unknown object_type 'unknown'",
        ),
        (
            "object_reference missing parent",
            lambda data: data["nodes"][2].pop("parent"),
            "Missing required string field 'parent'",
        ),
        (
            "object_reference missing prim_path",
            lambda data: data["nodes"][2].pop("prim_path"),
            "Missing required string field 'prim_path'",
        ),
        (
            "object_reference missing object_type",
            lambda data: data["nodes"][2].pop("object_type"),
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
            {"id": "background", "name": "background", "type": "background"},
            # Kept at index 2 (after its parent at index 1) so the bad-data mutation lambdas
            # below can address it, and so the order satisfies the upstream ordering contract.
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
