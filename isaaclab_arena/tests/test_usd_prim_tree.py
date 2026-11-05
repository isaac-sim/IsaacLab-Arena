# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for USD prim-tree loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.usd.prim_tree import load_stage_prim_tree, load_usd_physics_roots, load_usd_prim_tree


def _test_kitchen_physics_prim_trees(_) -> bool:
    """Requires Lightwheel Robocasa kitchen USD on disk (Docker dev image)."""
    usd_path = AssetSpec(
        id="lightwheel_robocasa_kitchen",
        registry_name="lightwheel_robocasa_kitchen",
        params={"layout_id": 1, "style_id": 1},
    ).resolve_usd_path()
    tree = load_usd_prim_tree(usd_path)
    records = {record.relative_path: record for record in tree}

    counter = records.get("counter_right_main_group/top_geometry")
    assert counter is not None, "counter_right_main_group/top_geometry missing from kitchen USD"
    assert counter.object_type == ObjectType.BASE
    assert counter.bounding_box_root is not None
    assert all(size > 0.0 for size in counter.bounding_box_root.size_m)
    assert counter.collider is not None
    assert counter.collider.enabled
    assert counter.collider.type_name == "Cube"

    fridge = records.get("fridge_main_group")
    assert fridge is not None, "fridge_main_group missing from kitchen USD"
    assert fridge.object_type == ObjectType.ARTICULATION
    assert "fridge_door_joint" in fridge.joint_names
    fridge_door_joint = next(record for record in tree if record.relative_path.endswith("/fridge_door_joint"))
    assert fridge_door_joint.joint is not None
    assert fridge_door_joint.joint.type_name == "PhysicsRevoluteJoint"
    assert fridge_door_joint.joint.limit_unit == "degrees"
    assert fridge_door_joint.joint.body_0_path == "fridge_main_group/Refrigerator032"
    assert fridge_door_joint.joint.body_1_path == "fridge_main_group/Refrigerator032_fridge_door"

    physics_roots = load_usd_physics_roots(usd_path)
    assert len(physics_roots) == 21
    assert all(object_type == ObjectType.ARTICULATION for object_type in physics_roots.values())
    return True


def test_kitchen_physics_prim_trees():
    assert run_function_with_persistent_simulation_app(_test_kitchen_physics_prim_trees)


def _test_prim_tree_spatial_and_rigid_body_metadata(_, meters_per_unit: float | None) -> bool:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.usda"
        stage = Usd.Stage.CreateNew(str(path))
        if meters_per_unit is not None:
            UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())

        fixed_body = UsdGeom.Xform.Define(stage, "/Asset/FixedBody").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(fixed_body).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(fixed_body).CreateMassAttr(2.0)

        moving_body = UsdGeom.Xform.Define(stage, "/Asset/MovingBody")
        moving_body.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdPhysics.RigidBodyAPI.Apply(moving_body.GetPrim())
        UsdPhysics.MassAPI.Apply(moving_body.GetPrim()).CreateDensityAttr(250.0)
        collider = UsdGeom.Cube.Define(stage, "/Asset/MovingBody/Collider")
        collider.AddScaleOp().Set(Gf.Vec3f(0.5, 1.0, 1.5))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        mesh_collider = UsdGeom.Mesh.Define(stage, "/Asset/MovingBody/MeshCollider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh_collider)
        UsdPhysics.MeshCollisionAPI.Apply(mesh_collider).CreateApproximationAttr("convexHull")
        PhysxSchema.PhysxCollisionAPI.Apply(mesh_collider).CreateContactOffsetAttr(0.01)
        PhysxSchema.PhysxCollisionAPI(mesh_collider).CreateRestOffsetAttr(0.002)
        PhysxSchema.PhysxRigidBodyAPI.Apply(moving_body.GetPrim()).CreateLinearDampingAttr(0.3)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/MovingBody/Hinge")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/FixedBody")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/MovingBody")])
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(-30.0)
        joint.CreateUpperLimitAttr(60.0)
        stage.GetRootLayer().Save()

        records = {record.relative_path: record for record in load_usd_prim_tree(str(path))}

    distance_scale = 0.01 if meters_per_unit is None else meters_per_unit
    fixed_record = records["FixedBody"]
    assert fixed_record.rigid_body is not None
    assert fixed_record.rigid_body.kinematic
    assert fixed_record.rigid_body.mass_kg == 2.0

    moving_record = records["MovingBody"]
    assert moving_record.rigid_body is not None
    assert moving_record.rigid_body.density_kg_m3 == 250.0
    assert moving_record.rigid_body.linear_damping == pytest.approx(0.3)
    assert moving_record.instance is not None
    assert not moving_record.instance.instance
    assert moving_record.bounding_box_root is not None
    expected_bounds_m = (distance_scale, 2.0 * distance_scale, 3.0 * distance_scale)
    assert moving_record.bounding_box_root.center_m == pytest.approx(expected_bounds_m)
    assert moving_record.bounding_box_root.size_m == pytest.approx(expected_bounds_m)

    collider_record = records["MovingBody/Collider"]
    assert collider_record.collider is not None
    assert collider_record.collider.type_name == "Cube"
    assert collider_record.collider.enabled
    assert collider_record.collider.rigid_body_owner_path == "MovingBody"
    assert collider_record.collider.approximation is None
    assert collider_record.collider.contact_offset_m is None
    assert collider_record.collider.rest_offset_m is None

    mesh_collider_record = records["MovingBody/MeshCollider"]
    assert mesh_collider_record.collider is not None
    assert mesh_collider_record.collider.type_name == "Mesh"
    assert mesh_collider_record.collider.approximation == "convexHull"
    assert mesh_collider_record.collider.contact_offset_m == pytest.approx(0.01 * distance_scale)
    assert mesh_collider_record.collider.rest_offset_m == pytest.approx(0.002 * distance_scale)

    joint_record = records["MovingBody/Hinge"]
    assert joint_record.joint is not None
    assert joint_record.joint.type_name == "PhysicsRevoluteJoint"
    assert joint_record.joint.axis == "Z"
    assert joint_record.joint.lower_limit == -30.0
    assert joint_record.joint.upper_limit == 60.0
    assert joint_record.joint.limit_unit == "degrees"
    assert joint_record.joint.body_0_path == "FixedBody"
    assert joint_record.joint.body_1_path == "MovingBody"
    return True


@pytest.mark.parametrize("meters_per_unit", [1.0, 0.01, 0.001, None], ids=["meters", "cm", "mm", "default-cm"])
def test_prim_tree_spatial_and_rigid_body_metadata(meters_per_unit):
    assert run_function_with_persistent_simulation_app(
        _test_prim_tree_spatial_and_rigid_body_metadata, meters_per_unit=meters_per_unit
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

    records = {record.relative_path: record for record in load_stage_prim_tree(stage, root.GetPrim())}
    assert set(records) == {"Instance", "Instance/Collider"}
    body = records["Instance"]
    assert body.object_type == ObjectType.RIGID
    assert body.instance.instance
    assert body.instance.instanceable
    assert body.instance.prototype_path is not None
    assert body.bounding_box_root.center_m == pytest.approx((1.0, 2.0, 3.0))
    proxy = records["Instance/Collider"]
    assert proxy.instance.instance_proxy
    assert proxy.instance.prototype_path == f"{body.instance.prototype_path}/Collider"
    assert proxy.collider.rigid_body_owner_path == "Instance"
    assert proxy.bounding_box_root.size_m == pytest.approx((2.0, 2.0, 2.0))
    selected_proxy = stage.GetPrimAtPath("/Asset/Instance/Collider")
    proxy_records = load_stage_prim_tree(stage, selected_proxy, include_physical_root=True)
    assert len(proxy_records) == 1
    assert proxy_records[0].collider.rigid_body_owner_path == "/Asset/Instance"
    return True


def test_prim_tree_composed_instances():
    assert run_function_with_persistent_simulation_app(_test_prim_tree_composed_instances)


def _test_prim_tree_bounds_include_geometry_purpose(_, purpose: str) -> bool:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    body = UsdGeom.Xform.Define(stage, "/Asset/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    geometry = UsdGeom.Xform.Define(stage, "/Asset/Body/Geometry")
    geometry.CreatePurposeAttr(purpose)
    collider = UsdGeom.Cube.Define(stage, "/Asset/Body/Geometry/Collider")
    collider.AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    records = {record.relative_path: record for record in load_stage_prim_tree(stage, root)}
    bounds = records["Body"].bounding_box_root
    assert bounds is not None, f"Body bounds omitted its {purpose} geometry"
    assert bounds.center_m == pytest.approx((4.0, 0.0, 0.0))
    assert bounds.size_m == pytest.approx((2.0, 2.0, 2.0))

    default_collider = UsdGeom.Cube.Define(stage, "/Asset/Body/DefaultCollider")
    UsdPhysics.CollisionAPI.Apply(default_collider.GetPrim())
    records = {record.relative_path: record for record in load_stage_prim_tree(stage, root)}
    bounds = records["Body"].bounding_box_root
    assert bounds.center_m == pytest.approx((2.0, 0.0, 0.0))
    assert bounds.size_m == pytest.approx((6.0, 2.0, 2.0))
    return True


@pytest.mark.parametrize("purpose", ["default", "guide", "proxy", "render"])
def test_prim_tree_bounds_include_geometry_purpose(purpose):
    assert run_function_with_persistent_simulation_app(_test_prim_tree_bounds_include_geometry_purpose, purpose=purpose)


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

    records = {
        record.relative_path: record
        for record in load_stage_prim_tree(stage, stage.GetPrimAtPath("/Asset/Body/Joints"))
    }
    assert set(records) == {"Joint", "WorldJoint"}
    assert records["Joint"].joint.body_0_path == "/ExternalBody"
    assert records["Joint"].joint.body_1_path == "/Asset/Body"
    assert records["WorldJoint"].joint.body_0_path is None

    records = {record.relative_path: record for record in load_stage_prim_tree(stage, body, include_physical_root=True)}
    assert records["Geometry/Collider"].collider.rigid_body_owner_path == "."
    assert records["Joints/Joint"].joint.body_1_path == "."
    assert records["Joints/Joint"].joint.body_0_path == "/ExternalBody"
    return True


def test_prim_tree_subtree_preserves_body_paths():
    assert run_function_with_persistent_simulation_app(_test_prim_tree_subtree_preserves_body_paths)
