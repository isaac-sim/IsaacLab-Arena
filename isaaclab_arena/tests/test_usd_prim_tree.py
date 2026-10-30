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
from isaaclab_arena.utils.usd.prim_tree import load_usd_physics_roots, load_usd_prim_tree


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


def _test_prim_tree_spatial_and_rigid_body_metadata(_) -> bool:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.usda"
        stage = Usd.Stage.CreateNew(str(path))
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

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/MovingBody/Hinge")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/FixedBody")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/MovingBody")])
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(-30.0)
        joint.CreateUpperLimitAttr(60.0)
        stage.GetRootLayer().Save()

        records = {record.relative_path: record for record in load_usd_prim_tree(str(path))}

    fixed_record = records["FixedBody"]
    assert fixed_record.rigid_body is not None
    assert fixed_record.rigid_body.kinematic
    assert fixed_record.rigid_body.mass_kg == 2.0

    moving_record = records["MovingBody"]
    assert moving_record.rigid_body is not None
    assert moving_record.rigid_body.density_kg_m3 == 250.0
    assert moving_record.bounding_box_root is not None
    assert moving_record.bounding_box_root.center_m == pytest.approx((1.0, 2.0, 3.0))
    assert moving_record.bounding_box_root.size_m == pytest.approx((1.0, 2.0, 3.0))

    collider_record = records["MovingBody/Collider"]
    assert collider_record.collider is not None
    assert collider_record.collider.type_name == "Cube"
    assert collider_record.collider.enabled
    assert collider_record.collider.rigid_body_owner_path == "MovingBody"
    assert collider_record.collider.approximation is None

    mesh_collider_record = records["MovingBody/MeshCollider"]
    assert mesh_collider_record.collider is not None
    assert mesh_collider_record.collider.type_name == "Mesh"
    assert mesh_collider_record.collider.approximation == "convexHull"

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


def test_prim_tree_spatial_and_rigid_body_metadata():
    assert run_function_with_persistent_simulation_app(_test_prim_tree_spatial_and_rigid_body_metadata)
