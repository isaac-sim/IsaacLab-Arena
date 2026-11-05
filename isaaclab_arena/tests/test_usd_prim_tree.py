# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for USD prim-tree loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.usd.prim_tree import (
    UsdBoundingBox,
    UsdColliderInfo,
    UsdInstanceInfo,
    UsdJointInfo,
    UsdPrimRecord,
    UsdRigidBodyInfo,
    format_usd_prim_tree,
    load_stage_prim_tree,
    load_usd_prim_tree,
    usd_prim_records_to_dicts,
)


@pytest.fixture
def detailed_body_record():
    expected = {
        "relative_path": "Case/Body",
        "object_type": "rigid",
        "joint_names": [],
        "bounding_box_root": {"center_m": [0.12345678901234566, 2.0, 3.0], "size_m": [1.0, 2.0, 3.0]},
        "joint": None,
        "collider": {
            "type_name": "Mesh",
            "enabled": True,
            "rigid_body_owner_path": "Case/Body",
            "approximation": "convexHull",
            "contact_offset_m": 0.012345678901234567,
            "rest_offset_m": None,
        },
        "rigid_body": {
            "enabled": True,
            "kinematic": False,
            "mass_kg": 1.2345678901234567,
            "density_kg_m3": None,
            "center_of_mass": [0.1, 0.2, 0.3],
            "diagonal_inertia": [1.0, 2.0, 3.0],
            "disable_gravity": False,
            "linear_damping": 0.3,
            "angular_damping": 0.4,
        },
        "instance": {
            "instance": False,
            "instance_proxy": True,
            "instanceable": False,
            "prototype_path": "/Prototype/Body",
        },
    }
    record = UsdPrimRecord(
        "Case/Body",
        ObjectType.RIGID,
        bounding_box_root=UsdBoundingBox(**expected["bounding_box_root"]),
        collider=UsdColliderInfo(**expected["collider"]),
        rigid_body=UsdRigidBodyInfo(**expected["rigid_body"]),
        instance=UsdInstanceInfo(**expected["instance"]),
    )
    return record, expected


def test_prim_tree_compact_text_preserves_hierarchy(detailed_body_record):
    body, _ = detailed_body_record
    joint = UsdJointInfo("PhysicsRevoluteJoint", True, "Case/Body", None, "Z", -30.0, 60.0, "degrees")
    records = [
        UsdPrimRecord("Case/Joints/lid", ObjectType.BASE, joint=joint),
        body,
        UsdPrimRecord("Case/Joints", ObjectType.BASE),
        UsdPrimRecord("Case", ObjectType.ARTICULATION, ("lid",)),
        UsdPrimRecord("Ground", ObjectType.BASE),
    ]
    assert (
        format_usd_prim_tree(records, heading="BACKGROUND PRIM TREE:")
        == "BACKGROUND PRIM TREE:\n"
        "Case (articulation joints=[lid])\n"
        "  Body (rigid collider(type=Mesh,approximation=convexHull))\n"
        "  Joints (base)\n"
        "    lid (base joint(type=PhysicsRevoluteJoint))\n"
        "Ground (base)"
    )
    assert records[0].relative_path == "Case/Joints/lid"


def test_prim_tree_json_preserves_metadata_and_precision(detailed_body_record):
    record, expected = detailed_body_record
    output = json.dumps(usd_prim_records_to_dicts([record]), allow_nan=False)
    assert json.loads(output) == [expected]


def test_prim_tree_json_encodes_non_finite_numbers():
    joint = UsdJointInfo("PhysicsRevoluteJoint", True, None, "Body", "Z", float("-inf"), float("inf"), "degrees")
    record = UsdPrimRecord(
        "Joint",
        ObjectType.BASE,
        joint=joint,
        bounding_box_root=UsdBoundingBox((float("nan"), float("inf"), float("-inf")), (1.0, 2.0, 3.0)),
    )
    output = json.loads(json.dumps(usd_prim_records_to_dicts([record]), allow_nan=False))[0]
    assert output["joint"] == {
        "type_name": "PhysicsRevoluteJoint",
        "enabled": True,
        "body_0_path": None,
        "body_1_path": "Body",
        "axis": "Z",
        "lower_limit": "-inf",
        "upper_limit": "inf",
        "limit_unit": "degrees",
    }
    assert output["bounding_box_root"] == {"center_m": ["nan", "inf", "-inf"], "size_m": [1.0, 2.0, 3.0]}
    assert record.joint.lower_limit == float("-inf")


def _records_by_path(records: list[UsdPrimRecord]) -> dict[str, UsdPrimRecord]:
    return {record.relative_path: record for record in records}


def _test_kitchen_physics_prim_trees(_) -> bool:
    """Requires Lightwheel Robocasa kitchen USD on disk (Docker dev image)."""
    usd_path = AssetSpec(
        id="lightwheel_robocasa_kitchen",
        registry_name="lightwheel_robocasa_kitchen",
        params={"layout_id": 1, "style_id": 1},
    ).resolve_usd_path()
    tree = load_usd_prim_tree(usd_path)
    records = _records_by_path(tree)

    counter = records.get("counter_right_main_group/top_geometry")
    assert counter is not None, "counter_right_main_group/top_geometry missing from kitchen USD"
    assert counter.object_type == ObjectType.BASE
    assert counter.bounding_box_root is not None
    assert all(size > 0.0 for size in counter.bounding_box_root.size_m)

    fridge = records.get("fridge_main_group")
    assert fridge is not None, "fridge_main_group missing from kitchen USD"
    assert fridge.object_type == ObjectType.ARTICULATION
    assert "fridge_door_joint" in fridge.joint_names
    fridge_door_joint = next(record for record in tree if record.relative_path.endswith("/fridge_door_joint"))
    assert fridge_door_joint.joint is not None
    assert fridge_door_joint.joint.body_0_path == "fridge_main_group/Refrigerator032"
    assert fridge_door_joint.joint.body_1_path == "fridge_main_group/Refrigerator032_fridge_door"
    return True


def test_kitchen_physics_prim_trees():
    assert run_function_with_persistent_simulation_app(_test_kitchen_physics_prim_trees)


def _test_prim_tree_spatial_metadata(_, meters_per_unit: float | None) -> bool:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.usda"
        stage = Usd.Stage.CreateNew(str(path))
        if meters_per_unit is not None:
            UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())

        fixed_body = UsdGeom.Xform.Define(stage, "/Asset/FixedBody").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(fixed_body)

        moving_body = UsdGeom.Xform.Define(stage, "/Asset/MovingBody")
        moving_body.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdPhysics.RigidBodyAPI.Apply(moving_body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(moving_body.GetPrim())
        mass.CreateMassAttr(1000.0)
        mass.CreateDensityAttr(2.0)
        collider = UsdGeom.Cube.Define(stage, "/Asset/MovingBody/Collider")
        collider.AddScaleOp().Set(Gf.Vec3f(0.5, 1.0, 1.5))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(collider.GetPrim()).CreateContactOffsetAttr(0.01)
        PhysxSchema.PhysxCollisionAPI(collider.GetPrim()).CreateRestOffsetAttr(0.002)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/MovingBody/Hinge")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/FixedBody")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/MovingBody")])
        stage.GetRootLayer().Save()

        records = _records_by_path(load_usd_prim_tree(str(path)))

    distance_scale = 0.01 if meters_per_unit is None else meters_per_unit
    moving_record = records["MovingBody"]
    assert moving_record.bounding_box_root is not None
    expected_bounds_m = (distance_scale, 2.0 * distance_scale, 3.0 * distance_scale)
    assert moving_record.bounding_box_root.center_m == pytest.approx(expected_bounds_m)
    assert moving_record.bounding_box_root.size_m == pytest.approx(expected_bounds_m)
    assert moving_record.rigid_body.mass_kg == pytest.approx(1.0)
    assert moving_record.rigid_body.density_kg_m3 == pytest.approx(2.0 * 0.001 / distance_scale**3)

    collider_record = records["MovingBody/Collider"]
    assert collider_record.collider is not None
    assert collider_record.collider.rigid_body_owner_path == "MovingBody"
    assert collider_record.collider.contact_offset_m == pytest.approx(0.01 * distance_scale)
    assert collider_record.collider.rest_offset_m == pytest.approx(0.002 * distance_scale)

    joint_record = records["MovingBody/Hinge"]
    assert joint_record.joint is not None
    assert joint_record.joint.body_0_path == "FixedBody"
    assert joint_record.joint.body_1_path == "MovingBody"
    return True


@pytest.mark.parametrize("meters_per_unit", [0.001, None], ids=["mm", "default-cm"])
def test_prim_tree_spatial_metadata(meters_per_unit):
    assert run_function_with_persistent_simulation_app(
        _test_prim_tree_spatial_metadata, meters_per_unit=meters_per_unit
    )


def _test_prim_tree_composed_instances(_) -> bool:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    template = UsdGeom.Xform.Define(stage, "/Template").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(template)
    collider = UsdGeom.Cube.Define(stage, "/Template/Collider")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    stage.SetDefaultPrim(root.GetPrim())
    instance = UsdGeom.Xform.Define(stage, "/Asset/Instance")
    instance.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    instance.GetPrim().GetReferences().AddInternalReference("/Template")
    instance.GetPrim().SetInstanceable(True)

    records = _records_by_path(load_stage_prim_tree(stage, root.GetPrim()))
    assert set(records) == {"Instance", "Instance/Collider"}
    body = records["Instance"]
    assert body.object_type == ObjectType.RIGID
    assert body.bounding_box_root.center_m == pytest.approx((1.0, 2.0, 3.0))
    proxy = records["Instance/Collider"]
    assert proxy.collider.rigid_body_owner_path == "Instance"
    assert proxy.bounding_box_root.size_m == pytest.approx((2.0, 2.0, 2.0))
    selected_proxy = stage.GetPrimAtPath("/Asset/Instance/Collider")
    proxy_records = load_stage_prim_tree(stage, selected_proxy, include_physical_root=True)
    assert len(proxy_records) == 1
    assert proxy_records[0].collider.rigid_body_owner_path == "/Asset/Instance"
    return True


def test_prim_tree_composed_instances():
    assert run_function_with_persistent_simulation_app(_test_prim_tree_composed_instances)


def _test_prim_tree_bounds_include_geometry_purposes(_) -> bool:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    purposes = ("default", "guide", "proxy", "render")
    for index, purpose in enumerate(purposes):
        body = UsdGeom.Xform.Define(stage, f"/Asset/{purpose}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        geometry = UsdGeom.Xform.Define(stage, f"/Asset/{purpose}/Geometry")
        geometry.CreatePurposeAttr(purpose)
        collider = UsdGeom.Cube.Define(stage, f"/Asset/{purpose}/Geometry/Collider")
        collider.AddTranslateOp().Set(Gf.Vec3d(4.0 * index, 0.0, 0.0))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    records = _records_by_path(load_stage_prim_tree(stage, root))
    for index, purpose in enumerate(purposes):
        bounds = records[purpose].bounding_box_root
        assert bounds is not None, f"Bounds omitted {purpose} geometry"
        assert bounds.center_m == pytest.approx((4.0 * index, 0.0, 0.0))
    return True


def test_prim_tree_bounds_include_geometry_purposes():
    assert run_function_with_persistent_simulation_app(_test_prim_tree_bounds_include_geometry_purposes)


def _test_prim_tree_subtree_preserves_body_paths(_) -> bool:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    body = UsdGeom.Xform.Define(stage, "/Asset/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    geometry = UsdGeom.Xform.Define(stage, "/Asset/Body/Geometry")
    geometry.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    collider = UsdGeom.Cube.Define(stage, "/Asset/Body/Geometry/Collider")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    external_body = UsdGeom.Xform.Define(stage, "/ExternalBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(external_body)
    external_collider = UsdGeom.Cube.Define(stage, "/ExternalBody/Collider")
    UsdPhysics.CollisionAPI.Apply(external_collider.GetPrim())
    joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/Body/Joints/Joint")
    joint.CreateBody0Rel().SetTargets([external_collider.GetPath()])
    joint.CreateBody1Rel().SetTargets([collider.GetPath()])
    world_joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/Body/Joints/WorldJoint")
    world_joint.CreateBody1Rel().SetTargets([body.GetPath()])

    records = load_stage_prim_tree(stage, geometry.GetPrim())
    assert [record.relative_path for record in records] == ["Collider"]
    assert records[0].collider.rigid_body_owner_path == "/Asset/Body"
    assert records[0].bounding_box_root.center_m == pytest.approx((0.0, 0.0, 0.0))

    records = _records_by_path(load_stage_prim_tree(stage, stage.GetPrimAtPath("/Asset/Body/Joints")))
    assert set(records) == {"Joint", "WorldJoint"}
    assert records["Joint"].joint.body_0_path == "/ExternalBody"
    assert records["Joint"].joint.body_1_path == "/Asset/Body"
    assert records["WorldJoint"].joint.body_0_path is None

    records = _records_by_path(load_stage_prim_tree(stage, body, include_physical_root=True))
    assert records["Geometry/Collider"].collider.rigid_body_owner_path == "."
    assert records["Joints/Joint"].joint.body_1_path == "."
    assert records["Joints/Joint"].joint.body_0_path == "/ExternalBody"
    return True


def test_prim_tree_subtree_preserves_body_paths():
    assert run_function_with_persistent_simulation_app(_test_prim_tree_subtree_preserves_body_paths)


def _test_absolute_subtree_without_default_prim(_) -> bool:
    from pxr import Usd, UsdGeom, UsdPhysics

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "no_default_prim.usda"
        stage = Usd.Stage.CreateNew(str(path))
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        stage.GetRootLayer().Save()

        records = load_usd_prim_tree(str(path), prim_path="/World/Body")
        assert len(records) == 1
        assert records[0].relative_path == "."
        assert records[0].object_type == ObjectType.RIGID

        with pytest.raises(AssertionError, match="no valid default prim"):
            load_usd_prim_tree(str(path))
        with pytest.raises(AssertionError, match="no valid default prim"):
            load_usd_prim_tree(str(path), prim_path="World/Body")
    return True


def test_absolute_subtree_without_default_prim():
    assert run_function_with_persistent_simulation_app(_test_absolute_subtree_without_default_prim)
