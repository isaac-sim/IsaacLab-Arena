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
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
    UnresolvedArenaEnvGraphSpec,
)
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
)
from isaaclab_arena.environments.graph_spec_utils import relation_class_for_spatial_constraint_type
from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On, PositionLimits

TEST_DATA_DIR = Path(__file__).parent / "test_data"
_INIT_GRAPH = TEST_DATA_DIR / "pick_and_place_maple_table_init_env_graph.yaml"
_FULL_GRAPH = TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml"


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
    assert task.kind == "PickAndPlaceTask"
    assert TaskRegistry().is_registered(task.kind)
    assert task.initial_state_spec_id == "state_spec_0"
    assert task.success_state_spec_id == "state_spec_1"
    assert task.params["pick_up_object"] == "rubiks_cube_hot3d_robolab"
    assert task.params["destination_location"] == "bowl_ycb_robolab"

    second_task = spec.tasks_by_id["pick_and_place_1"]
    assert second_task.kind == "PickAndPlaceTask"
    assert second_task.initial_state_spec_id == "state_spec_1"
    assert second_task.success_state_spec_id == "state_spec_2"
    assert second_task.params["pick_up_object"] == "mug_ycb_robolab"

    initial_state = spec.state_specs_by_id["state_spec_0"]
    assert isinstance(initial_state, ArenaEnvGraphStateSpec)
    assert len(initial_state.spatial_constraints) == 6

    cube_limits = initial_state.spatial_constraints[2]
    assert cube_limits.kind == "position_limits"
    assert cube_limits.subject == "rubiks_cube_hot3d_robolab"
    assert cube_limits.reference is None
    assert cube_limits.params == {"x_min": 0.55, "x_max": 0.70, "y_min": -0.40, "y_max": -0.10}

    initial_mug_position = initial_state.spatial_constraints[5]
    assert initial_mug_position.kind == "at_position"
    assert initial_mug_position.subject == "mug_ycb_robolab"
    assert initial_mug_position.reference is None
    assert initial_mug_position.params == {"x": 0.65, "y": 0.25, "z": 0.85}

    # Derived states are differential: state_spec_1 records only the cube's relocation, not a
    # full snapshot (the unchanged constraints are inherited from state_spec_0).
    success_state = spec.state_specs_by_id["state_spec_1"]
    assert len(success_state.spatial_constraints) == 1
    cube_on_bowl = success_state.spatial_constraints[0]
    assert cube_on_bowl.kind == "on"
    assert cube_on_bowl.reference == "bowl_ycb_robolab"
    assert cube_on_bowl.subject == "rubiks_cube_hot3d_robolab"

    table_anchor = initial_state.spatial_constraints[0]
    assert table_anchor.kind == "is_anchor"
    assert table_anchor.subject == "maple_table_robolab_table"
    assert table_anchor.reference is None
    assert relation_class_for_spatial_constraint_type(table_anchor.kind) is IsAnchor
    assert relation_class_for_spatial_constraint_type(cube_limits.kind) is PositionLimits
    assert relation_class_for_spatial_constraint_type(initial_mug_position.kind) is AtPosition
    assert relation_class_for_spatial_constraint_type(cube_on_bowl.kind) is On


def test_arena_env_graph_spec_parses_at_position():
    data = _minimal_env_graph_data()
    data["state_specs"][0]["spatial_constraints"] = [_at_position_constraint()]

    spec = ArenaEnvGraphSpec.from_dict(data)
    assert spec.tasks_by_id["task_0"].kind == "PickAndPlaceTask"
    state_spec = spec.state_specs_by_id["state_0"]
    fixed_position = state_spec.spatial_constraints[0]

    assert fixed_position.kind == "at_position"
    assert fixed_position.subject == "cube"
    assert fixed_position.reference is None
    assert fixed_position.params == {"x": 0.1, "y": 0.2, "z": 0.3}


def test_arena_env_graph_spec_validate_rejects_mutated_missing_reference():
    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    spec.state_specs[0].spatial_constraints[0].subject = "missing_table"

    with pytest.raises(AssertionError, match="unknown subject node 'missing_table'"):
        spec.validate()


def test_arena_env_graph_spec_validate_rejects_mutated_invalid_relationship_shape():
    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    constraint = spec.state_specs[0].spatial_constraints[0]
    constraint.kind = "on"

    with pytest.raises(AssertionError, match="requires relation.reference"):
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

    # from_dict runs the model_validator; Pydantic wraps the assertion in a ValidationError.
    with pytest.raises(ValidationError, match="targets unknown node 'missing_node'"):
        ArenaEnvGraphSpec.from_dict(data)


def test_validate_rejects_duplicate_cli_override_args():
    data = _minimal_env_graph_data()
    data["cli_override_specs"] = [
        {"arg": "object", "target_node_id": "cube"},
        {"arg": "object", "target_node_id": "robot"},
    ]

    with pytest.raises(ValidationError, match="Duplicate cli_override arg '--object'"):
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
            "missing spatial subject",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].pop("subject"),
            "subject",
        ),
        (
            "binary relationship missing reference",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].__setitem__("kind", "on"),
            "requires relation.reference",
        ),
        (
            "unary relationship with reference",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].__setitem__("reference", "cube"),
            "must not define relation.reference",
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
            "unknown task kind",
            lambda data: data["tasks"][0].__setitem__("kind", "UnknownTask"),
            "Unknown task kind 'UnknownTask'",
        ),
        (
            "unknown spatial constraint kind",
            lambda data: data["state_specs"][0]["spatial_constraints"][0].__setitem__("kind", "unknown"),
            "Unknown relation kind 'unknown'",
        ),
    ]

    for label, mutate, error_match in cases:
        data = _minimal_env_graph_data()
        mutate(data)

        with pytest.raises(ValidationError) as exc_info:
            ArenaEnvGraphSpec.from_dict(data)
        assert error_match in str(exc_info.value), label


# --- ArenaEnvGraphSpec.resolve_constraints: chaining a partially-wired graph into a full one ---


def test_resolve_constraints_reproduces_groundtruth_full_graph():
    """Resolving the partial init graph yields the hand-authored full graph (structurally)."""
    resolved = UnresolvedArenaEnvGraphSpec.from_yaml(_INIT_GRAPH).resolve()
    groundtruth = ArenaEnvGraphSpec.from_yaml(_FULL_GRAPH)

    # N tasks yield N+1 state specs, chained 0..N.
    assert set(resolved.state_specs_by_id) == set(groundtruth.state_specs_by_id)
    assert len(resolved.state_specs) == len(resolved.tasks) + 1

    # Each state's spatial constraints match content-wise (IDs are auto-generated and may differ
    # from hand-authored ones, so compare as sets of (kind, subject, reference, params) tuples).
    for state_id, groundtruth_state in groundtruth.state_specs_by_id.items():
        got = resolved.state_specs_by_id[state_id]
        assert _spatial_contents(got) == _spatial_contents(groundtruth_state), f"spatial mismatch in {state_id}"
        assert got.is_delta == groundtruth_state.is_delta, f"is_delta mismatch in {state_id}"

    # The initial state is a full snapshot; every derived state is a delta off its predecessor.
    assert resolved.state_specs_by_id["state_spec_0"].is_delta is False
    assert all(resolved.state_specs_by_id[f"state_spec_{i}"].is_delta for i in range(1, len(resolved.tasks) + 1))

    # Tasks are wired correctly (compared in order; auto-generated IDs differ from hand-authored ones).
    assert len(resolved.tasks) == len(groundtruth.tasks)
    for got_task, groundtruth_task in zip(resolved.tasks, groundtruth.tasks):
        assert got_task.kind == groundtruth_task.kind
        assert got_task.initial_state_spec_id == groundtruth_task.initial_state_spec_id
        assert got_task.success_state_spec_id == groundtruth_task.success_state_spec_id
        assert got_task.params == groundtruth_task.params


def test_unresolved_graph_is_not_directly_loadable():
    """The init graph leaves tasks unwired, so the strict ArenaEnvGraphSpec loader rejects it."""
    with pytest.raises(ValidationError):
        ArenaEnvGraphSpec.from_yaml(_INIT_GRAPH)


def test_chain_wires_each_success_state_as_next_initial_state():
    """task[i].success_state_spec_id == task[i+1].initial_state_spec_id (a single chain)."""
    resolved = UnresolvedArenaEnvGraphSpec.from_yaml(_INIT_GRAPH).resolve()
    ordered = resolved.tasks  # preserved in execution order by resolve()
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.success_state_spec_id == later.initial_state_spec_id


def test_task_without_a_transition_is_rejected():
    """A task whose class declares no success_state_transition fails loudly rather than silently skipping."""
    spec = UnresolvedArenaEnvGraphSpec.from_yaml(_INIT_GRAPH)
    spec.tasks[0].kind = "NoTask"  # registered, but declares no transition
    with pytest.raises(NotImplementedError, match="success_state_transition not implemented"):
        spec.resolve()


def _spatial_contents(state: ArenaEnvGraphStateSpec) -> set[tuple]:
    """Project a state's spatial constraints to a set of (kind, subject, reference, params) tuples.

    ID-independent, so auto-generated constraint IDs from resolve() compare correctly against
    hand-authored IDs in the groundtruth YAML.
    """
    return {(c.kind, c.subject, c.reference, tuple(sorted(c.params.items()))) for c in state.spatial_constraints}


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
            "initial_state_spec_id": "state_0",
            "success_state_spec_id": "state_0",
            "kind": "PickAndPlaceTask",
            "params": {},
        }],
        "state_specs": [{
            "id": "state_0",
            "spatial_constraints": [{
                "id": "table_is_anchor",
                "kind": "is_anchor",
                "subject": "table",
            }],
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
        "kind": "at_position",
        "subject": "cube",
        "params": {"x": x, "y": y, "z": z},
    }
