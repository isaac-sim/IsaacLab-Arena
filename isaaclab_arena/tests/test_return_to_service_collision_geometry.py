# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check composed collider geometry and separation independently of robot motion."""

import math
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

import pytest

from isaaclab_arena_environments.return_to_service.collision_geometry import (
    CollisionBoxes,
    pairwise_box_separation,
    read_collision_boxes,
    transform_boxes,
)


def _boxes(centers, extents, axes=None):
    centers = torch.as_tensor(centers, dtype=torch.float64).reshape(-1, 3)
    if axes is None:
        axes = torch.eye(3, dtype=centers.dtype).expand(len(centers), 3, 3)
    return CollisionBoxes(
        centers,
        torch.as_tensor(axes, dtype=centers.dtype),
        torch.as_tensor(extents, dtype=centers.dtype).expand_as(centers),
        ("feature",) * len(centers),
    )


def _stage(path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def _cube(stage, path="/Asset/collider", *, enabled=True):
    from pxr import UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(enabled)
    cube.GetPrim().SetCustomDataByKey("arena:feature", "guide")
    return cube


def test_transform_maps_noncentered_boxes_and_preserves_local_geometry():
    boxes = _boxes(((1, 0, 0), (0, 2, 0)), (0.1, 0.2, 0.3))
    # A deliberately nonunit XYZW quaternion still represents a quarter turn.
    pose = torch.tensor((3, 4, 5, 0, 0, math.sqrt(2), math.sqrt(2)), dtype=torch.float64)
    result = transform_boxes(boxes, pose)
    torch.testing.assert_close(result.centers, pose.new_tensor(((3, 5, 5), (1, 4, 5))))
    torch.testing.assert_close(result.axes[0], pose.new_tensor(((0, 1, 0), (-1, 0, 0), (0, 0, 1))))
    torch.testing.assert_close(boxes.centers, pose.new_tensor(((1, 0, 0), (0, 2, 0))))
    assert result.half_extents is boxes.half_extents and result.features == boxes.features


def test_pairwise_separation_distinguishes_gap_contact_and_overlap():
    left = _boxes(((0, 0, 0),), (0.5, 0.5, 0.5))
    right = _boxes(((2, 0, 0), (1, 0, 0), (0.9, 0, 0)), (0.5, 0.5, 0.5))
    separation = pairwise_box_separation(left, right)
    torch.testing.assert_close(separation, left.centers.new_tensor(((1, 0, -0.1),)))
    torch.testing.assert_close(separation.T, pairwise_box_separation(right, left))


def test_edge_cross_axes_separate_skew_boxes_whose_face_projections_overlap():
    pose = torch.tensor(
        (-0.5765011240, 0.3804832869, -0.2554200747, -0.7198719507, -0.4809437944, -0.1020370155, 0.4899651913),
        dtype=torch.float64,
    )
    left = transform_boxes(_boxes(((0, 0, 0),), (0.8, 0.15, 0.1)), pose)
    right = _boxes(((0, 0, 0),), (0.25, 0.45, 0.2))
    assert pairwise_box_separation(left, right).item() > 0
    face_normals = torch.cat((left.axes[0], right.axes[0]))
    delta = right.centers[0] - left.centers[0]
    radius_left = ((face_normals @ left.axes[0].T).abs() * left.half_extents[0]).sum(-1)
    radius_right = ((face_normals @ right.axes[0].T).abs() * right.half_extents[0]).sum(-1)
    assert bool(((face_normals @ delta).abs() <= radius_left + radius_right).all())


def test_separation_agrees_with_independent_halfspace_feasibility():
    rng = np.random.default_rng(743)
    seen = set()
    for _ in range(30):
        centers = rng.uniform(-0.7, 0.7, size=(2, 3))
        axes = Rotation.random(2, random_state=rng).as_matrix()
        extents = rng.uniform(0.1, 0.6, size=(2, 3))
        left = _boxes(centers[0:1], extents[0], axes[0:1])
        right = _boxes(centers[1:2], extents[1], axes[1:2])
        constraints, limits = [], []
        for center, basis, half in zip(centers, axes, extents):
            constraints.extend((basis, -basis))
            limits.extend((basis @ center + half, -basis @ center + half))
        solution = linprog(
            np.zeros(3),
            A_ub=np.concatenate(constraints),
            b_ub=np.concatenate(limits),
            bounds=[(None, None)] * 3,
            method="highs",
        )
        assert solution.status in (0, 2), solution.message
        intersects = solution.status == 0
        assert (pairwise_box_separation(left, right).item() <= 0) == intersects
        seen.add(intersects)
    assert seen == {False, True}, "The geometric fixtures must exercise both outcomes."


def test_reader_excludes_root_pose_but_preserves_parent_transforms_spawn_scale_and_invisible_colliders(tmp_path):
    from pxr import UsdGeom

    path = tmp_path / "part.usda"
    stage, root = _stage(path)
    root.AddTranslateOp().Set((12, 13, 14))
    root.AddRotateXOp().Set(90)
    parent = UsdGeom.Xform.Define(stage, "/Asset/assembly")
    parent.AddTranslateOp().Set((1, 2, 3))
    parent.AddRotateZOp().Set(90)
    cube = _cube(stage, "/Asset/assembly/collider")
    cube.AddTranslateOp().Set((0.1, 0.2, 0.3))
    cube.AddScaleOp().Set((0.2, 0.4, 0.6))
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    _cube(stage, "/Asset/disabled", enabled=False).AddTranslateOp().Set((999, 0, 0))
    UsdGeom.Cube.Define(stage, "/Asset/visual_only")
    stage.GetRootLayer().Save()
    boxes = read_collision_boxes(path, scale=(2, 2, 2), dtype=torch.float64)
    assert boxes.features == ("guide",)
    torch.testing.assert_close(boxes.centers, torch.tensor(((1.6, 4.2, 6.6),), dtype=torch.float64))
    torch.testing.assert_close(boxes.half_extents, torch.tensor(((0.4, 0.8, 1.2),), dtype=torch.float64))
    expected = torch.tensor((((0, 1, 0), (-1, 0, 0), (0, 0, 1)),), dtype=torch.float64)
    torch.testing.assert_close(boxes.axes, expected)


def test_reader_uses_the_custom_composed_spawn_asset(tmp_path):
    from pxr import UsdGeom

    source = tmp_path / "source.usda"
    source_stage, _ = _stage(source)
    _cube(source_stage)
    source_stage.GetRootLayer().Save()
    overlay = tmp_path / "physics.usda"
    stage, root = _stage(overlay)
    root.GetPrim().GetReferences().AddReference("source.usda")
    UsdGeom.Cube(stage.GetPrimAtPath("/Asset/collider")).GetSizeAttr().Set(0.4)
    stage.GetRootLayer().Save()
    boxes = read_collision_boxes(overlay)
    torch.testing.assert_close(boxes.half_extents, torch.full((1, 3), 0.2))
    assert read_collision_boxes(source).half_extents[0, 0] == 1


@pytest.mark.parametrize("kind", ("root_transform", "child_transform", "size", "collision", "body", "order"))
def test_reader_rejects_single_time_samples_that_override_default_geometry(tmp_path, kind):
    from pxr import UsdPhysics

    path = tmp_path / "sampled.usda"
    stage, root = _stage(path)
    cube = _cube(stage)
    if kind == "root_transform":
        root.AddScaleOp().Set((2, 2, 2), 1)
    elif kind == "child_transform":
        cube.AddTranslateOp().Set((1, 2, 3), 1)
    elif kind == "size":
        cube.GetSizeAttr().Set(5, 1)
    elif kind == "collision":
        UsdPhysics.CollisionAPI(cube.GetPrim()).GetCollisionEnabledAttr().Set(False, 1)
    elif kind == "body":
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).GetRigidBodyEnabledAttr().Set(False, 1)
    else:
        cube.AddTranslateOp().Set((1, 2, 3))
        cube.GetXformOpOrderAttr().Set([], 1)
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="Animated"):
        read_collision_boxes(path)


def test_reader_rejects_time_sampled_scale_above_default_prim(tmp_path):
    from pxr import UsdGeom

    path = tmp_path / "sampled_ancestor.usda"
    stage, root = _stage(path)
    parent = UsdGeom.Xform.Define(stage, "/Asset/part")
    stage.SetDefaultPrim(parent.GetPrim())
    _cube(stage, "/Asset/part/collider")
    root.AddScaleOp().Set((2, 2, 2), 1)
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="Animated frame ancestry"):
        read_collision_boxes(path)


@pytest.mark.parametrize("axis", ("X", "Y", "Z"))
def test_reader_rejects_unequal_cylinder_radial_scale(tmp_path, axis):
    from pxr import UsdGeom, UsdPhysics

    path = tmp_path / "elliptical.usda"
    stage, _ = _stage(path)
    cylinder = UsdGeom.Cylinder.Define(stage, "/Asset/collider")
    cylinder.CreateAxisAttr(axis)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    cylinder.GetPrim().SetCustomDataByKey("arena:feature", "shaft")
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="Unequal radial"):
        read_collision_boxes(path, scale=(1, 2, 3))


@pytest.mark.parametrize(
    "kind",
    ("articulation", "child_articulation", "child_body", "joint", "shear", "root_scale", "root_reflection", "reset"),
)
def test_reader_rejects_geometry_requiring_a_different_motion_model(tmp_path, kind):
    from pxr import UsdPhysics

    path = tmp_path / "unsupported.usda"
    stage, root = _stage(path)
    cube = _cube(stage)
    scale = (1, 1, 1)
    if kind == "articulation":
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    elif kind == "child_articulation":
        UsdPhysics.ArticulationRootAPI.Apply(cube.GetPrim())
    elif kind == "child_body":
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    elif kind == "joint":
        UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joint")
    elif kind == "root_scale":
        root.AddScaleOp().Set((2, 1, 1))
    elif kind == "root_reflection":
        root.AddScaleOp().Set((-1, 1, 1))
    elif kind == "reset":
        cube.SetResetXformStack(True)
    else:
        cube.AddRotateZOp().Set(45)
        scale = (2, 1, 1)
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError):
        read_collision_boxes(path, scale=scale)


def test_reader_rejects_missing_enabled_geometry_and_unlabeled_shapes(tmp_path):
    from pxr import UsdPhysics

    path = tmp_path / "part.usda"
    stage, _ = _stage(path)
    cube = _cube(stage, enabled=False)
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="no enabled collision"):
        read_collision_boxes(path)
    UsdPhysics.CollisionAPI(cube.GetPrim()).GetCollisionEnabledAttr().Set(True)
    cube.GetPrim().ClearCustomDataByKey("arena:feature")
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="feature name"):
        read_collision_boxes(path)


@pytest.mark.parametrize("axis, dimensions", (("X", (0.6, 0.2, 0.2)), ("Y", (0.2, 0.6, 0.2)), ("Z", (0.2, 0.2, 0.6))))
def test_reader_uses_physical_cylinder_dimensions_for_every_axis(tmp_path, axis, dimensions):
    from pxr import UsdGeom, UsdPhysics

    path = tmp_path / "cylinder.usda"
    stage, _ = _stage(path)
    cylinder = UsdGeom.Cylinder.Define(stage, "/Asset/collider")
    cylinder.CreateRadiusAttr(0.1)
    cylinder.CreateHeightAttr(0.6)
    cylinder.CreateAxisAttr(axis)
    cylinder.CreateExtentAttr(((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)))
    cylinder.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    cylinder.GetPrim().SetCustomDataByKey("arena:feature", "shaft")
    stage.GetRootLayer().Save()
    boxes = read_collision_boxes(path, dtype=torch.float64)
    torch.testing.assert_close(boxes.half_extents[0], torch.tensor(dimensions, dtype=torch.float64) / 2)


def test_reader_ignores_stale_cube_display_extent(tmp_path):
    from pxr import UsdGeom

    path = tmp_path / "cube.usda"
    stage, _ = _stage(path)
    cube = _cube(stage)
    cube.CreateExtentAttr(((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)))
    cube.CreatePurposeAttr(UsdGeom.Tokens.proxy)
    stage.GetRootLayer().Save()
    torch.testing.assert_close(read_collision_boxes(path).half_extents, torch.ones((1, 3)))


@pytest.mark.parametrize("kind", ("parent_transform", "primitive_dimension"))
def test_reader_rejects_animated_geometry_before_caching(tmp_path, kind):
    from pxr import Usd

    path = tmp_path / "animated.usda"
    stage, root = _stage(path)
    cube = _cube(stage)
    attribute = root.AddTranslateOp().GetAttr() if kind == "parent_transform" else cube.GetSizeAttr()
    for frame in (0, 1):
        value = (frame, 0, 0) if kind == "parent_transform" else float(frame + 1)
        attribute.Set(value, Usd.TimeCode(frame))
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="Animated"):
        read_collision_boxes(path)


def test_reader_rejects_mesh_cooking_that_can_exceed_visual_bounds(tmp_path):
    from pxr import UsdGeom, UsdPhysics

    path = tmp_path / "mesh.usda"
    stage, _ = _stage(path)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/collider")
    mesh.CreatePointsAttr(((-1, -0.1, 0), (1, -0.1, 0), (0, 0.1, 0)))
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("boundingSphere")
    mesh.GetPrim().SetCustomDataByKey("arena:feature", "mesh")
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="cube/cylinder"):
        read_collision_boxes(path)


@pytest.mark.parametrize("kind", ("collision", "child_body"))
def test_reader_rejects_animated_enablement_that_would_omit_live_geometry(tmp_path, kind):
    from pxr import Usd, UsdPhysics

    path = tmp_path / "enablement.usda"
    stage, _ = _stage(path)
    _cube(stage)
    changing = _cube(stage, "/Asset/changing")
    if kind == "collision":
        attribute = UsdPhysics.CollisionAPI(changing.GetPrim()).GetCollisionEnabledAttr()
    else:
        attribute = UsdPhysics.RigidBodyAPI.Apply(changing.GetPrim()).CreateRigidBodyEnabledAttr(False)
    attribute.Set(False)
    attribute.Set(False, Usd.TimeCode(0))
    attribute.Set(True, Usd.TimeCode(1))
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="Animated.*enablement"):
        read_collision_boxes(path)
