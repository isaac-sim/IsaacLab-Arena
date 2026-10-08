# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check object geometry against native spawn settings and rigid-body pose frames."""

import numpy as np
import trimesh


def _write_box_mesh(stage, path):
    from pxr import UsdGeom

    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(cube.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(cube.faces))
    mesh.CreateFaceVertexIndicesAttr().Set(cube.faces.reshape(-1).tolist())
    return mesh


def test_rigid_geometry_uses_body_pose_frame(tmp_path):
    from isaaclab.sim import UsdFileCfg
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    source_path = tmp_path / "nested.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    root = UsdGeom.Xform.Define(source_stage, "/Asset").GetPrim()
    source_stage.SetDefaultPrim(root)
    group = UsdGeom.Xform.Define(source_stage, "/Asset/Group")
    group.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    body = UsdGeom.Xform.Define(source_stage, "/Asset/Group/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    body.AddRotateZOp().Set(90.0)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = _write_box_mesh(source_stage, "/Asset/Group/Body/Cube")
    mesh.AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 3.0))
    source_stage.GetRootLayer().Save()
    original_content = source_path.read_bytes()

    obj = Object(
        name="nested",
        object_type=ObjectType.RIGID,
        spawner_cfg=UsdFileCfg(usd_path=str(source_path), scale=(2.0, 2.0, 2.0)),
    )
    bounds = obj.get_bounding_box()
    np.testing.assert_allclose(bounds.min_point, [[-2.0, -4.0, -6.0]], atol=1e-6)
    np.testing.assert_allclose(bounds.max_point, [[2.0, 4.0, 6.0]], atol=1e-6)
    collision_mesh = obj.get_collision_mesh()
    np.testing.assert_allclose(collision_mesh.bounds, [[-2.0, -4.0, -6.0], [2.0, 4.0, 6.0]])
    collision_mesh.vertices[:] = 0.0
    np.testing.assert_allclose(obj.get_collision_mesh().extents, [4.0, 8.0, 12.0])
    assert obj.get_collision_mesh(excluded_prim_paths=("/Asset/Group",)) is None
    assert obj.get_contact_sensor_prim_path() == obj.get_prim_path() + "/Group/Body"
    assert source_path.read_bytes() == original_content


def test_mesh_exclusions_preserve_source_instances(tmp_path):
    from isaaclab.sim import UsdFileCfg
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    geometry_path = tmp_path / "geometry.usda"
    geometry_stage = Usd.Stage.CreateNew(str(geometry_path))
    geometry_root = UsdGeom.Xform.Define(geometry_stage, "/Geometry").GetPrim()
    geometry_stage.SetDefaultPrim(geometry_root)
    for name, position in (("Keep", 0.0), ("Exclude", 10.0)):
        mesh = _write_box_mesh(geometry_stage, f"/Geometry/{name}")
        mesh.AddTranslateOp().Set(Gf.Vec3d(position, 0.0, 0.0))
    geometry_stage.GetRootLayer().Save()

    source_path = tmp_path / "instances.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/Asset").GetPrim()
    source_stage.SetDefaultPrim(source_root)
    instance = UsdGeom.Xform.Define(source_stage, "/Asset/Instance").GetPrim()
    instance.GetReferences().AddReference("geometry.usda")
    instance.SetInstanceable(True)
    source_stage.GetRootLayer().Save()
    original_content = source_path.read_bytes()

    obj = Object(name="background", object_type=ObjectType.BASE, spawner_cfg=UsdFileCfg(usd_path=str(source_path)))
    excluded_mesh = obj.get_collision_mesh(excluded_prim_paths=("/Asset/Instance/Exclude",))
    np.testing.assert_allclose(excluded_mesh.bounds, [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(obj.get_collision_mesh().bounds, [[-1.0, -1.0, -1.0], [11.0, 1.0, 1.0]])
    assert instance.IsInstance()
    assert source_stage.GetPrimAtPath("/Asset/Instance/Exclude").IsInstanceProxy()
    assert source_path.read_bytes() == original_content
