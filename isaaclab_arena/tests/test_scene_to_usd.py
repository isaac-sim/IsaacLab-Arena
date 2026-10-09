# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pathlib

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
EPS = 1e-6


def _test_scene_to_usd(simulation_app, output_path: pathlib.Path) -> bool:

    from pxr import Gf, Usd

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    # Set up a test scene
    asset_registry = AssetRegistry()

    kitchen = asset_registry.get_asset_by_name("kitchen")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    kitchen_initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_xyzw=(0, 0, -0.70711, 0.70711))
    kitchen.set_initial_pose(kitchen_initial_pose)
    cracker_box_initial_pose = Pose(position_xyz=(0.4, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    cracker_box.set_initial_pose(cracker_box_initial_pose)

    # Composed scene
    scene = Scene(assets=[kitchen, cracker_box])

    # Save the scene to a USD file on disk
    print(f"Saving scene to {output_path}")
    scene.export_to_usd(output_path)

    # Load the USD file and check that the scene was saved correctly
    stage = Usd.Stage.Open(output_path.as_posix())
    root_prim = stage.GetDefaultPrim()
    assert root_prim.GetPath() == "/World"

    test_prim_names = [kitchen.name, cracker_box.name]
    test_prim_poses = {
        kitchen.name: kitchen_initial_pose,
        cracker_box.name: cracker_box_initial_pose,
    }

    # Function to convert a pxr.Gf.Quatf to a numpy array
    def to_numpy_q_xyzw(q: Gf.Quatf) -> np.ndarray:
        return np.array([*q.GetImaginary(), q.GetReal()])

    # Loop over all the prims and check that the scene was saved correctly
    assert len(root_prim.GetChildren()) == len(test_prim_names)
    for prim in root_prim.GetChildren():
        prim_name = prim.GetName()
        assert prim_name in test_prim_names
        print(f"Checking prim: {prim_name}")
        prim_position = prim.GetAttribute("xformOp:translate").Get()
        prim_orientation = prim.GetAttribute("xformOp:orient").Get()
        assert np.linalg.norm(prim_position - test_prim_poses[prim_name].position_xyz) < EPS
        assert np.linalg.norm(to_numpy_q_xyzw(prim_orientation) - test_prim_poses[prim_name].rotation_xyzw) < EPS
        print(f"Prim {prim_name} position: {prim_position}")
        print(f"Prim {prim_name} orientation: {to_numpy_q_xyzw(prim_orientation)}")
        print(f"Prim {prim_name} expected position: {test_prim_poses[prim_name].position_xyz}")
        print(f"Prim {prim_name} expected orientation: {test_prim_poses[prim_name].rotation_xyzw}")

    return True


def test_scene_to_usd(tmp_path: pathlib.Path):
    # The passed tmp_path is a directory.
    output_path = tmp_path / "saved_kitchen_with_cracker_box_for_test.usd"
    result = run_function_with_persistent_simulation_app(
        _test_scene_to_usd,
        headless=HEADLESS,
        output_path=output_path,
    )
    assert result, "Test failed"


def test_scene_export_uses_native_usd_spawn_config(tmp_path: pathlib.Path):
    from isaaclab.sim import MassPropertiesCfg, UsdFileCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    source_path = tmp_path / "variant_source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = source_stage.DefinePrim("/Source", "Xform")
    source_stage.SetDefaultPrim(source_root)
    UsdPhysics.RigidBodyAPI.Apply(source_root)
    UsdPhysics.MassAPI.Apply(source_root).CreateMassAttr(1.0)
    shape_variants = source_root.GetVariantSets().AddVariantSet("shape")
    for variant_name, size in (("small", 0.2), ("large", 0.8)):
        shape_variants.AddVariant(variant_name)
        shape_variants.SetVariantSelection(variant_name)
        with shape_variants.GetVariantEditContext():
            cube = UsdGeom.Cube.Define(source_stage, "/Source/Cube")
            cube.CreateSizeAttr(size)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    shape_variants.SetVariantSelection("small")
    source_stage.GetRootLayer().Save()

    obj = Object(
        name="box",
        object_type=ObjectType.RIGID,
        spawn_cfg=UsdFileCfg(
            usd_path=str(source_path),
            variants={"shape": "large"},
            scale=(2.0, 1.0, 1.0),
            mass_props=MassPropertiesCfg(mass=3.0),
            activate_contact_sensors=True,
        ),
        initial_pose=Pose(position_xyz=(1.0, 2.0, 3.0), rotation_xyzw=(0.0, 0.0, 1.0, 0.0)),
    )
    # Export must use the current native configuration, including changes made after construction.
    obj.spawn_cfg.scale = (3.0, 2.0, 1.0)
    spawn_settings = obj.spawn_cfg.to_dict()
    output_path = tmp_path / "singleton.usda"
    Scene([obj]).export_to_usd(output_path, root_prim_path="/Export")

    exported_stage = Usd.Stage.Open(str(output_path))
    exported_root = exported_stage.GetPrimAtPath("/Export/box")
    np.testing.assert_allclose(exported_root.GetAttribute("xformOp:translate").Get(), (1.0, 2.0, 3.0))
    np.testing.assert_allclose(exported_root.GetAttribute("xformOp:scale").Get(), (3.0, 2.0, 1.0))
    exported_rotation = exported_root.GetAttribute("xformOp:orient").Get()
    np.testing.assert_allclose((*exported_rotation.GetImaginary(), exported_rotation.GetReal()), (0.0, 0.0, 1.0, 0.0))
    assert UsdGeom.Cube(exported_stage.GetPrimAtPath("/Export/box/Cube")).GetSizeAttr().Get() == pytest.approx(0.8)
    assert UsdPhysics.MassAPI(exported_root).GetMassAttr().Get() == pytest.approx(3.0)
    assert exported_root.GetAttribute("physxContactReport:threshold").Get() == 0.0
    assert obj.spawn_cfg.to_dict() == spawn_settings
    assert source_root.GetVariantSet("shape").GetVariantSelection() == "small"


def test_scene_export_supports_native_procedural_objects(tmp_path: pathlib.Path):
    from isaaclab.sim import CollisionBaseCfg, CuboidCfg, MassPropertiesCfg, RigidBodyBaseCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.scene import Scene

    obj = Object(
        name="box",
        object_type=ObjectType.RIGID,
        spawn_cfg=CuboidCfg(
            size=(0.2, 0.4, 0.6),
            rigid_props=RigidBodyBaseCfg(),
            collision_props=CollisionBaseCfg(),
            mass_props=MassPropertiesCfg(mass=2.0),
            activate_contact_sensors=True,
        ),
    )
    obj.object_cfg.init_state.pos = (1.0, 2.0, 3.0)
    output_path = tmp_path / "procedural.usda"
    Scene([obj]).export_to_usd(output_path)

    exported_stage = Usd.Stage.Open(str(output_path))
    exported_root = exported_stage.GetPrimAtPath("/World/box")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(exported_root)
    np.testing.assert_allclose(bounds.ComputeAlignedRange().GetMin(), (0.9, 1.8, 2.7), atol=EPS)
    np.testing.assert_allclose(bounds.ComputeAlignedRange().GetMax(), (1.1, 2.2, 3.3), atol=EPS)
    assert UsdPhysics.MassAPI(exported_root).GetMassAttr().Get() == pytest.approx(2.0)
    assert exported_root.GetAttribute("physxContactReport:threshold").Get() == 0.0


def test_scene_export_requires_single_asset_spawner(tmp_path: pathlib.Path):
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.scene import Scene

    source_path = tmp_path / "member.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/Member").GetPrim()
    source_stage.SetDefaultPrim(source_root)
    UsdPhysics.RigidBodyAPI.Apply(source_root)
    cube = UsdGeom.Cube.Define(source_stage, "/Member/Cube")
    cube.CreateSizeAttr(0.2)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    source_stage.GetRootLayer().Save()
    member = Object(name="member", object_type=ObjectType.RIGID, usd_path=str(source_path))
    obj = RigidObjectSet(name="box", objects=[member, member])
    with pytest.raises(AssertionError, match="Select a concrete member"):
        Scene([obj]).export_to_usd(tmp_path / "ambiguous.usda")


@pytest.mark.parametrize("reset_xform_stack", [False, True])
def test_scene_export_places_nested_rigid_body_in_its_pose_frame(tmp_path: pathlib.Path, reset_xform_stack: bool):
    from isaaclab.sim import UsdFileCfg
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    source_path = tmp_path / "nested_body.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_stage.SetDefaultPrim(UsdGeom.Xform.Define(source_stage, "/Source").GetPrim())
    parent = UsdGeom.Xform.Define(source_stage, "/Source/Offset")
    parent.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    body = UsdGeom.Xform.Define(source_stage, "/Source/Offset/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(2.0, 3.0, 0.0))
    body.AddRotateZOp().Set(90.0)
    body.SetResetXformStack(reset_xform_stack)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    cube = UsdGeom.Cube.Define(source_stage, "/Source/Offset/Body/Cube")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3d(0.2, 0.4, 0.6))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    source_stage.GetRootLayer().Save()
    source_contents = source_stage.GetRootLayer().ExportToString()

    pose = Pose(position_xyz=(4.0, 5.0, 6.0), rotation_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5))
    obj = Object(
        name="nested",
        object_type=ObjectType.RIGID,
        spawn_cfg=UsdFileCfg(usd_path=str(source_path), scale=(2.0, 3.0, 1.0)),
        initial_pose=pose,
    )
    output_path = tmp_path / "nested_export.usda"
    Scene([obj]).export_to_usd(output_path)

    exported_stage = Usd.Stage.Open(str(output_path))
    exported_body = exported_stage.GetPrimAtPath("/World/nested/Offset/Body")
    body_transform = Gf.Transform(UsdGeom.Xformable(exported_body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    np.testing.assert_allclose(body_transform.GetTranslation(), pose.position_xyz, atol=EPS)
    rotation = body_transform.GetRotation().GetQuat()
    np.testing.assert_allclose((*rotation.GetImaginary(), rotation.GetReal()), pose.rotation_xyzw, atol=EPS)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(exported_body)
    half_extents = np.array((0.2, 0.1, 0.3) if reset_xform_stack else (0.4, 0.3, 0.3))
    np.testing.assert_allclose(
        bounds.ComputeAlignedRange().GetMin(), np.array(pose.position_xyz) - half_extents, atol=EPS
    )
    np.testing.assert_allclose(
        bounds.ComputeAlignedRange().GetMax(), np.array(pose.position_xyz) + half_extents, atol=EPS
    )
    placement_bounds = obj.get_world_bounding_box()
    np.testing.assert_allclose(bounds.ComputeAlignedRange().GetMin(), placement_bounds.min_point[0].numpy(), atol=EPS)
    np.testing.assert_allclose(bounds.ComputeAlignedRange().GetMax(), placement_bounds.max_point[0].numpy(), atol=EPS)
    assert source_stage.GetRootLayer().ExportToString() == source_contents
