# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for separating cabinet frames and their authored joints."""

import numpy as np

import pytest

from isaaclab_arena_environments.workbench_cleanup.scripts.fixed_articulations import create_fixed_articulations
from isaaclab_arena_environments.workbench_cleanup.scripts.mesh_partition import (
    open_mesh_partition_stage,
    partition_mesh,
)


@pytest.fixture
def cabinet_source(tmp_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "cabinet.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Cabinet")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root.AddTranslateOp().Set(Gf.Vec3d(1, 2, 0))
    base = UsdGeom.Xform.Define(stage, "/Cabinet/Base")
    base.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.3))
    base.AddRotateZOp().Set(20)
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Cabinet/Base/Mesh")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.04, 0, 0))
    points = []
    for y in (0, 2, 4):
        points.extend([(0, y, 0), (1, y, 0), (0, y, 1)])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr(list(range(9)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    for name in ("Left", "Right"):
        body = UsdGeom.Xform.Define(stage, f"/Cabinet/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.6, 0.7))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        end = UsdGeom.Cube.Define(stage, f"/Cabinet/{name}/Geometry")
        end.AddTranslateOp().Set(Gf.Vec3d(0.02, 0, 0))
        UsdPhysics.CollisionAPI.Apply(end.GetPrim())
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Cabinet/Joints/{name}")
        joint.CreateBody0Rel().SetTargets([mesh.GetPath()])
        joint.CreateBody1Rel().SetTargets([end.GetPath()])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.01)
        joint.CreateUpperLimitAttr(0.4)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.11, 0.12, 0.13))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.01, 0.02, 0.03))
    stage.GetRootLayer().Save()
    return path


def _groups():
    return {name: {"frame_mesh": f"Base/Mesh/{name}Frame", "joints": [f"Joints/{name}"]} for name in ("Left", "Right")}


def test_fixed_articulations_preserve_world_geometry_and_joint_frames(cabinet_source, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    before = cabinet_source.read_bytes()
    stage = open_mesh_partition_stage(str(cabinet_source))
    partition_mesh(stage, "Base/Mesh", {"LeftFrame": [0], "RightFrame": [1]})
    cache = UsdGeom.XformCache()
    transforms = {}
    for name in ("Left", "Right"):
        for suffix in (f"Base/Mesh/{name}Frame", f"{name}/Geometry", "Base/Mesh"):
            transforms[suffix] = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/Cabinet/{suffix}")))
    reports = create_fixed_articulations(stage, _groups())
    output = tmp_path / "partitioned.usdc"
    stage.GetRootLayer().Export(str(output))
    stage = Usd.Stage.Open(str(output))
    cache = UsdGeom.XformCache()
    assert len(reports) == 2
    assert not stage.GetPrimAtPath("/Cabinet/Base").HasAPI(UsdPhysics.RigidBodyAPI)
    assert not stage.GetPrimAtPath("/Cabinet").HasAPI(UsdPhysics.ArticulationRootAPI)
    for name in ("Left", "Right"):
        prefix = f"/Cabinet/Articulations/{name}"
        assert stage.GetPrimAtPath(prefix).HasAPI(UsdPhysics.ArticulationRootAPI)
        assert stage.GetPrimAtPath(prefix + "/Frame").HasAPI(UsdPhysics.RigidBodyAPI)
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prefix)),
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prefix + "/Frame")),
            atol=1e-6,
        )
        for old, new in (
            (f"Base/Mesh/{name}Frame", "Frame/Geometry"),
            (f"{name}/Geometry", f"{name}/Geometry"),
            ("Base/Mesh", "Frame"),
        ):
            np.testing.assert_allclose(
                cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{prefix}/{new}")), transforms[old], atol=1e-6
            )
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"{prefix}/Joints/{name}"))
        assert list(joint.GetBody0Rel().GetTargets()) == [prefix + "/Frame"]
        assert list(joint.GetBody1Rel().GetTargets()) == [prefix + f"/{name}/Geometry"]
        assert joint.GetAxisAttr().Get() == "X"
        assert joint.GetUpperLimitAttr().Get() == pytest.approx(0.4)
        assert list(joint.GetLocalPos0Attr().Get()) == pytest.approx([0.11, 0.12, 0.13])
        assert list(joint.GetLocalPos1Attr().Get()) == pytest.approx([0.01, 0.02, 0.03])
        assert not stage.GetPrimAtPath(f"/Cabinet/Joints/{name}").IsActive()
        assert not stage.GetPrimAtPath(f"/Cabinet/{name}").IsActive()
    assert cabinet_source.read_bytes() == before


def test_fixed_articulations_reject_animated_ancestors(cabinet_source):
    from pxr import Gf, UsdGeom

    stage = open_mesh_partition_stage(str(cabinet_source))
    partition_mesh(stage, "Base/Mesh", {"LeftFrame": [0], "RightFrame": [1]})
    base = UsdGeom.Xformable(stage.GetPrimAtPath("/Cabinet/Base"))
    base.GetOrderedXformOps()[0].Set(Gf.Vec3d(1, 2, 3), 1.0)
    base.GetOrderedXformOps()[0].Set(Gf.Vec3d(2, 2, 3), 2.0)
    with pytest.raises(AssertionError, match="Animated body transforms"):
        create_fixed_articulations(stage, _groups())


def test_fixed_articulations_reject_unselected_connections(cabinet_source):
    stage = open_mesh_partition_stage(str(cabinet_source))
    partition_mesh(stage, "Base/Mesh", {"LeftFrame": [0], "RightFrame": [1]})
    with pytest.raises(AssertionError, match="Unselected joint"):
        create_fixed_articulations(stage, {"Left": _groups()["Left"]})
