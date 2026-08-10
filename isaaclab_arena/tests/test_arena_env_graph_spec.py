# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`~isaaclab_arena.environment_spec.arena_env_graph_spec.ArenaEnvGraphSpec`."""

import subprocess
import yaml
from pathlib import Path

import pytest
from pydantic import ValidationError

import isaaclab_arena_environments
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry, TaskRegistry
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import CliOverrideSpec, TaskCompositionType
from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On, PositionLimitsBox, PositionLimitsCylindrical
from isaaclab_arena.tests.utils.constants import TestConstants

TEST_DATA_DIR = Path(__file__).parent / "test_data"
_GRAPH = TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml"
_OBJECT_SET_GRAPH = TEST_DATA_DIR / "object_set_maple_table_env_graph.yaml"
_DEBUG_VIEW_GRAPH = TEST_DATA_DIR / "placement_debug_view_env_graph.yaml"


def test_graph_spec_loads_pick_and_place_yaml():
    spec = ArenaEnvGraphSpec.from_yaml(_GRAPH)

    assert spec.env_name == "pick_and_place_maple_table_default"
    assert spec.embodiment.registry_name == "droid_abs_joint_pos"
    assert spec.background.registry_name == "maple_table_robolab"
    assert len(spec.objects) == 3
    assert len(spec.object_references) == 1
    assert len(spec.relations) == 6
    assert len(spec.task.subtasks) == 2
    assert spec.task.composition is TaskCompositionType.SEQUENTIAL

    table = spec.object_references[0]
    assert table.id == "maple_table_robolab_table"
    assert table.parent_id == "maple_table_robolab"
    assert table.prim_path == "{ENV_REGEX_NS}/maple_table_robolab/table"
    assert table.object_type == ObjectType.RIGID

    mug = next(obj for obj in spec.objects if obj.id == "mug_ycb_robolab")
    assert mug.registry_name == "mug_ycb_robolab"

    task = spec.task.subtasks[0]
    assert task.kind == "PickAndPlaceTask"
    assert TaskRegistry().is_registered(task.kind)
    assert task.params["pick_up_object"] == "rubiks_cube_hot3d_robolab"
    assert task.params["destination_location"] == "bowl_ycb_robolab"

    second_task = spec.task.subtasks[1]
    assert second_task.kind == "PickAndPlaceTask"
    assert second_task.params["pick_up_object"] == "mug_ycb_robolab"

    cube_limits = spec.relations[2]
    assert cube_limits.kind == "position_limits_box"
    assert cube_limits.subject == "rubiks_cube_hot3d_robolab"
    assert cube_limits.reference is None
    assert cube_limits.params == {
        "x_min": 0.55,
        "x_max": 0.7,
        "y_min": -0.4,
        "y_max": -0.1,
    }

    mug_position = spec.relations[5]
    assert mug_position.kind == "at_position"
    assert mug_position.subject == "mug_ycb_robolab"
    assert mug_position.reference is None
    assert mug_position.params == {"x": 0.65, "y": 0.25, "z": 0.85}

    table_anchor = spec.relations[0]
    assert table_anchor.kind == "is_anchor"
    assert table_anchor.subject == "maple_table_robolab_table"
    assert ObjectRelationLibraryRegistry().get_object_relation_by_name(table_anchor.kind) is IsAnchor
    assert ObjectRelationLibraryRegistry().get_object_relation_by_name(cube_limits.kind) is PositionLimitsBox
    assert ObjectRelationLibraryRegistry().get_object_relation_by_name(mug_position.kind) is AtPosition
    assert ObjectRelationLibraryRegistry().get_object_relation_by_name(spec.relations[1].kind) is On


def test_graph_spec_parses_radial_position_limits():
    """Graph specs preserve cylindrical position-limit parameters for relation construction."""

    data = _minimal_env_graph_data()
    data["relations"] = [{
        "kind": "position_limits_cylindrical",
        "subject": "cube",
        "params": {"center_x": 0.5, "center_y": -0.25, "radius_min": 0.1, "radius_max": 0.3},
    }]

    (relation_spec,) = ArenaEnvGraphSpec.from_dict(data).relations
    relation_class = ObjectRelationLibraryRegistry().get_object_relation_by_name(relation_spec.kind)
    relation = relation_class(**relation_spec.params)

    assert isinstance(relation, PositionLimitsCylindrical)
    assert relation.center_x == 0.5
    assert relation.center_y == -0.25
    assert relation.radius_min == 0.1
    assert relation.radius_max == 0.3


def test_graph_spec_accepts_legacy_position_limits_box_name():
    """The legacy YAML relation name continues to construct a box relation."""

    data = _minimal_env_graph_data()
    data["relations"] = [{
        "kind": "position_limits",
        "subject": "cube",
        "params": {"x_min": 0.1, "x_max": 0.3},
    }]

    (relation_spec,) = ArenaEnvGraphSpec.from_dict(data).relations
    relation_class = ObjectRelationLibraryRegistry().get_object_relation_by_name(relation_spec.kind)
    relation = relation_class(**relation_spec.params)

    assert relation_class is PositionLimitsBox
    assert isinstance(relation, PositionLimitsBox)


def test_graph_spec_loads_object_set_yaml():
    spec = ArenaEnvGraphSpec.from_yaml(_OBJECT_SET_GRAPH)

    assert len(spec.object_sets) == 1
    object_set = spec.object_sets[0]
    assert object_set.id == "pick_up_object_set"
    assert object_set.members == ["sweet_potato", "jug"]
    assert object_set.random_choice
    assert object_set.params == {}

    # An object set is one scene node: relations and task params address it by id like an object.
    on_relation = next(relation for relation in spec.relations if relation.subject == "pick_up_object_set")
    assert on_relation.kind == "on"
    assert on_relation.reference == "maple_table_robolab"
    assert spec.task.subtasks[0].params["pick_up_object"] == "pick_up_object_set"


def test_graph_spec_rejects_object_set_id_colliding_with_object():
    data = _minimal_env_graph_data()
    data["object_sets"] = [{"id": "cube", "members": ["sweet_potato", "jug"]}]
    with pytest.raises(ValidationError, match="Duplicate graph asset ids"):
        ArenaEnvGraphSpec.from_dict(data)


def test_graph_spec_rejects_object_set_without_valid_members():
    data = _minimal_env_graph_data()
    data["object_sets"] = [{"id": "variants", "members": []}]
    with pytest.raises(ValidationError, match="at least 1 item"):
        ArenaEnvGraphSpec.from_dict(data)

    data["object_sets"] = [{"id": "variants", "members": ["not_a_real_asset"]}]
    with pytest.raises(ValidationError, match="Unknown asset registry_name"):
        ArenaEnvGraphSpec.from_dict(data)


def test_graph_spec_rejects_non_rigid_object_set_member():
    data = _minimal_env_graph_data()
    data["object_sets"] = [{"id": "variants", "members": ["sweet_potato", "ground_plane"]}]
    with pytest.raises(ValidationError, match="must be a rigid object"):
        ArenaEnvGraphSpec.from_dict(data)


def test_graph_spec_rejects_object_set_params_shadowing_spec_fields():
    data = _minimal_env_graph_data()
    data["object_sets"] = [{"id": "variants", "members": ["sweet_potato"], "params": {"objects": []}}]
    with pytest.raises(ValidationError, match="params must not set"):
        ArenaEnvGraphSpec.from_dict(data)


def test_graph_spec_rejects_relation_referencing_unknown_object_set():
    data = _minimal_env_graph_data()
    data["object_sets"] = [{"id": "variants", "members": ["sweet_potato", "jug"]}]
    data["relations"].append({"kind": "on", "subject": "other_variants", "reference": "table"})
    with pytest.raises(ValidationError, match="unknown subject"):
        ArenaEnvGraphSpec.from_dict(data)


def test_graph_spec_parses_at_position():

    data = _minimal_env_graph_data()
    data["relations"] = [{
        "kind": "at_position",
        "subject": "cube",
        "params": {"x": 0.1, "y": 0.2, "z": 0.3},
    }]
    spec = ArenaEnvGraphSpec.from_dict(data)
    relation = spec.relations[0]
    assert relation.kind == "at_position"
    assert relation.subject == "cube"
    assert relation.reference is None
    assert relation.params == {"x": 0.1, "y": 0.2, "z": 0.3}


def test_cli_override_specs_parsed_from_yaml():
    spec = ArenaEnvGraphSpec.from_yaml(_GRAPH)
    overrides = {override.arg: override for override in spec.cli_override_specs}
    assert overrides["embodiment"].target_node_id == "droid_abs_joint_pos"
    assert overrides["object"].target_node_id == "rubiks_cube_hot3d_robolab"


def test_add_cli_override_args_registers_declared_flags():
    import argparse

    from isaaclab_arena_environments.cli import add_cli_override_args

    parser = argparse.ArgumentParser()
    specs = ArenaEnvGraphSpec.read_cli_override_specs(_GRAPH)
    add_cli_override_args(parser, specs)

    args = parser.parse_args(["--object", "dex_cube"])
    assert args.object == "dex_cube"
    assert args.embodiment is None


def test_apply_cli_override_args_swaps_declared_target_registry_names():
    import argparse

    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    spec.cli_override_specs = [
        CliOverrideSpec(arg="object", target_node_id="cube"),
        CliOverrideSpec(arg="embodiment", target_node_id="robot"),
    ]
    spec.apply_cli_override_args(argparse.Namespace(object="dex_cube", embodiment="franka_ik"))
    assert spec.objects[0].registry_name == "dex_cube"
    assert spec.embodiment.registry_name == "franka_ik"


def test_apply_cli_override_args_leaves_unset_flags_as_authored():
    import argparse

    spec = ArenaEnvGraphSpec.from_dict(_minimal_env_graph_data())
    spec.cli_override_specs = [CliOverrideSpec(arg="object", target_node_id="cube")]
    spec.apply_cli_override_args(argparse.Namespace(object=None))
    assert spec.objects[0].registry_name == "rubiks_cube_hot3d_robolab"


def test_validate_rejects_cli_override_targeting_unknown_asset():
    data = _minimal_env_graph_data()
    data["cli_override_specs"] = [{"arg": "object", "target_node_id": "missing_node"}]
    with pytest.raises(ValidationError, match="targets unknown or non-swappable asset"):
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


def test_graph_spec_rejects_invalid_data():
    cases = [
        (
            "duplicate object id",
            lambda data: data["objects"].append(
                {"id": "cube", "registry_name": "rubiks_cube_hot3d_robolab", "params": {}}
            ),
            "Duplicate graph asset ids",
        ),
        (
            "binary relationship missing reference",
            lambda data: data["relations"].append({"kind": "on", "subject": "cube"}),
            "requires relation.reference",
        ),
        (
            "unary relationship with reference",
            lambda data: data["relations"][0].update({"reference": "background"}),
            "must not define relation.reference",
        ),
        (
            "unknown object reference parent",
            lambda data: data["object_references"].append({
                "id": "orphan_table",
                "parent_id": "missing",
                "prim_path": "/World/table",
                "object_type": "rigid",
            }),
            "references invalid parent",
        ),
        (
            "unknown task kind",
            lambda data: data["task"]["subtasks"][0].update({"kind": "UnknownTask"}),
            "Unknown task kind 'UnknownTask'",
        ),
        (
            "unknown spatial relation kind",
            lambda data: data["relations"][0].update({"kind": "unknown"}),
            "Unknown relation kind 'unknown'",
        ),
        (
            "unknown registry name",
            lambda data: data["objects"][0].update({"registry_name": "not_a_real_asset"}),
            "Unknown asset registry_name",
        ),
        (
            "atomic composition with two subtasks",
            lambda data: (
                data["task"].update({"composition": "atomic"}),
                data["task"]["subtasks"].append({
                    "kind": "PickAndPlaceTask",
                    "params": {
                        "pick_up_object": "cube",
                        "destination_location": "background",
                        "background_scene": "background",
                    },
                }),
            ),
            "composition 'atomic' requires exactly one atomic task",
        ),
        (
            "parallel composition with one subtask",
            lambda data: data["task"].update({"composition": "parallel"}),
            "composition 'parallel' requires at least two atomic tasks",
        ),
    ]

    for label, mutate, error_match in cases:
        data = _minimal_env_graph_data()
        mutate(data)
        with pytest.raises(ValidationError) as exc_info:
            ArenaEnvGraphSpec.from_dict(data)
        assert error_match in str(exc_info.value), label


def test_graph_spec_rejects_unknown_relation_subject():
    data = ArenaEnvGraphSpec.from_yaml(_GRAPH).to_dict()
    data["relations"][0]["subject"] = "missing_node"
    with pytest.raises(ValidationError, match="unknown subject"):
        ArenaEnvGraphSpec.from_dict(data)


def test_graph_spec_accepts_missing_object_reference_prim_path():
    data = ArenaEnvGraphSpec.from_yaml(_GRAPH).to_dict()
    data["object_references"][0].pop("prim_path", None)
    spec = ArenaEnvGraphSpec.from_dict(data)
    assert spec.object_references[0].prim_path is None


@pytest.mark.parametrize("empty_reference", ["", "   "])
def test_graph_spec_normalizes_empty_unary_relation_reference(empty_reference):
    data = _minimal_env_graph_data()
    data["relations"][0]["reference"] = empty_reference

    spec = ArenaEnvGraphSpec.from_dict(data)

    assert spec.relations[0].reference is None


def _minimal_env_graph_data():
    return {
        "env_name": "minimal_env_graph",
        "embodiment": {"id": "robot", "registry_name": "franka_ik"},
        "background": {"id": "background", "registry_name": "maple_table_robolab"},
        "objects": [{"id": "cube", "registry_name": "rubiks_cube_hot3d_robolab"}],
        "object_references": [{
            "id": "table",
            "parent_id": "background",
            "prim_path": "{ENV_REGEX_NS}/background/table",
            "object_type": "rigid",
        }],
        "relations": [{"kind": "is_anchor", "subject": "table"}],
        "task": {
            "composition": "atomic",
            "description": "pick up the cube",
            "subtasks": [{
                "kind": "PickAndPlaceTask",
                "params": {
                    "pick_up_object": "cube",
                    "destination_location": "cube",
                    "background_scene": "background",
                },
            }],
        },
    }


def test_graph_spec_accepts_missing_optional_fields():
    data = _minimal_env_graph_data()
    del data["object_references"]
    data["relations"] = [{"kind": "is_anchor", "subject": "background"}]
    data["task"]["subtasks"][0]["params"]["destination_location"] = "background"
    spec = ArenaEnvGraphSpec.from_dict(data)
    assert spec.object_references is None
    assert spec.object_sets is None
    assert spec.cli_override_specs is None


def test_graph_spec_omits_empty_optional_fields_from_dict():
    data = _minimal_env_graph_data()
    del data["object_references"]
    data["relations"] = [{"kind": "is_anchor", "subject": "background"}]
    data["task"]["subtasks"][0]["params"]["destination_location"] = "background"
    data["object_sets"] = []
    spec = ArenaEnvGraphSpec.from_dict(data)
    dumped = spec.to_dict()
    assert "object_references" not in dumped
    assert "object_sets" not in dumped
    assert "cli_override_specs" not in dumped
    assert spec.cli_override_specs is None


_REPLAY_SCRIPT = """
import sys

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

spec = ArenaEnvGraphSpec.from_yaml(sys.argv[1])
print("USD_PATH=" + spec.objects[0].resolve_usd_path())
"""


def _load_spec_in_a_fresh_process(path: Path) -> subprocess.CompletedProcess:
    """Load a written spec in a new process, where nothing has been registered by a search."""
    return subprocess.run(
        [TestConstants.python_path, "-c", _REPLAY_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_a_searched_simready_object_loads_in_a_fresh_process(tmp_path):
    usd_path = "s3://bucket/replay_kettle.usd"
    data = _minimal_env_graph_data()
    # How the agent writes a searched asset: the generic SimReady entry, which every process has,
    # carrying the USD path the search found. The asset is never opened to resolve it, so a path
    # that is only reachable from the runner still loads here.
    data["objects"] = [{"id": "cube", "registry_name": "simready_usd_object", "params": {"usd_path": usd_path}}]
    path = tmp_path / "generated_env_graph.yaml"
    ArenaEnvGraphSpec.from_dict(data).write_yaml(path)

    result = _load_spec_in_a_fresh_process(path)

    assert result.returncode == 0, f"loading the generated spec failed:\n{result.stderr}"
    assert f"USD_PATH={usd_path}" in result.stdout


def test_the_generic_simready_asset_is_rejected_as_an_object_set_member():
    data = _minimal_env_graph_data()
    data["object_sets"] = [{"id": "kettles", "members": ["simready_usd_object"]}]

    # It is rigid, so it passes the member type check; what it cannot do is arrive with a usd_path.
    with pytest.raises(ValidationError, match="cannot be an object set member"):
        ArenaEnvGraphSpec.from_dict(data)


def test_a_spec_naming_a_searched_simready_asset_by_its_search_name_is_rejected(tmp_path):
    data = _minimal_env_graph_data()
    data["objects"] = [{"id": "cube", "registry_name": "simready_replay_teapot", "params": {}}]
    path = tmp_path / "unregistered_env_graph.yaml"
    # A search name only exists in the process that searched, which is what every spec generated
    # before the agent started inlining the USD path looks like.
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = _load_spec_in_a_fresh_process(path)

    assert result.returncode != 0
    assert "Unknown asset registry_name 'simready_replay_teapot'" in result.stderr


def test_graph_spec_leaves_placement_debug_view_off_by_default():
    """A graph YAML that says nothing about debug visualization builds placement params with it off."""
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_checks_for_placer_params

    params = build_checks_for_placer_params(ArenaEnvGraphSpec.from_yaml(_GRAPH))

    assert not params.debug_visualize
    assert params.debug_visualize_output_path is None


def test_graph_spec_forwards_placement_debug_view_to_placer_params():
    """A YAML asking for the debug view reaches the params ObjectPlacer reads, both fields intact."""
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_checks_for_placer_params

    params = build_checks_for_placer_params(ArenaEnvGraphSpec.from_yaml(_DEBUG_VIEW_GRAPH))

    assert params.debug_visualize
    assert params.debug_visualize_output_path == "/tmp/placement_debug_view.rrd"


def test_graph_spec_leaves_shipped_envs_out_of_the_debug_view():
    """The debug view spawns a window on every build, so no env under version control may ask for it."""
    asking_for_the_view = [
        yaml_path
        for yaml_path in Path(isaaclab_arena_environments.__file__).parent.rglob("*.yaml")
        if "debug_visualize: true" in yaml_path.read_text()
    ]

    assert not asking_for_the_view, f"debug_visualize must stay off in shipped envs: {asking_for_the_view}"
