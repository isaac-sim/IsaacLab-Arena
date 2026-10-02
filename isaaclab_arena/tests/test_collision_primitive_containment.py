# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify solid-shape containment without filling empty corners of an asset's bounds."""

import itertools
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

import pytest

from isaaclab_arena.geometry.collision_geometry import CollisionBoxes, CollisionPrimitives, read_collision_primitives
from isaaclab_arena.geometry.containment import RegionContainment


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("axial", (0, 1, 2))
def test_curved_support_matches_independently_sampled_cylinder_surfaces(seed, axial):
    rng = np.random.default_rng(seed)
    primitive, world = Rotation.random(2, random_state=rng)
    center, translation = rng.uniform(-0.3, 0.3, (2, 3))
    radius, half_height = 0.027, 0.013
    extents = np.full(3, radius)
    extents[axial] = half_height
    boxes = CollisionBoxes(
        torch.tensor(center[None]),
        torch.tensor(primitive.as_matrix().T[None]),
        torch.tensor(extents[None]),
        ("cylinder",),
    )
    shape = CollisionPrimitives(boxes, torch.tensor([axial]))
    result = shape.bounds_in_frame(torch.tensor(np.r_[translation, world.as_quat()])).numpy()
    angles = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
    radial_axes = [index for index in range(3) if index != axial]
    rim = np.zeros((len(angles), 3))
    rim[:, radial_axes[0]] = radius * np.cos(angles)
    rim[:, radial_axes[1]] = radius * np.sin(angles)
    upper, lower = rim.copy(), rim.copy()
    upper[:, axial], lower[:, axial] = half_height, -half_height
    points = world.apply(primitive.apply(np.concatenate((upper, lower))) + center) + translation
    sampled = np.stack((points.min(axis=0), points.max(axis=0)))
    assert np.all(result[0] <= sampled[0] + 1e-14)
    assert np.all(result[1] >= sampled[1] - 1e-14)
    np.testing.assert_allclose(result, sampled, atol=2.1e-9, rtol=0)


def _write_shapes(path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cylinder = UsdGeom.Cylinder.Define(stage, "/Asset/Plug")
    cylinder.CreateAxisAttr("X")
    cylinder.CreateRadiusAttr(0.0105)
    cylinder.CreateHeightAttr(0.012)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.011))
    cube = UsdGeom.Cube.Define(stage, "/Asset/Tab")
    cube.AddTranslateOp().Set(Gf.Vec3d(0.015, 0, 0.015))
    cube.AddScaleOp().Set(Gf.Vec3f(0.011, 0.010, 0.004))
    for shape, feature in ((cylinder, "plug"), (cube, "tab")):
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        shape.GetPrim().SetCustomDataByKey("arena:feature", feature)
    stage.GetRootLayer().Save()


def test_compound_tilted_part_uses_occupied_geometry_not_an_empty_bounding_corner(tmp_path):
    path = tmp_path / "part.usda"
    _write_shapes(path)
    shapes = read_collision_primitives(path, dtype=torch.float64)
    rotation = Rotation.from_euler("y", 0.48)
    pose = torch.tensor(np.r_[np.zeros(3), rotation.as_quat()])
    exact_bottom = float(shapes.bounds_in_frame(pose)[0, 2])
    pose[2] = -exact_bottom
    scene = SimpleNamespace(part=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=None)))
    region = SimpleNamespace(center_xyz=(0, 0, 0.05), rotation_xyzw=(0, 0, 0, 1), half_extents_xyz=(0.1, 0.1, 0.05))
    checker = RegionContainment(scene, {"bin": region})
    assert bool(checker.contains("part", pose, "bin"))
    # A whole-asset box combines the tab's far X with the plug's lowest Z,
    # even though no physical component occupies that corner.
    corners = np.array(list(itertools.product((-0.006, 0.026), (-0.0105, 0.0105), (0.0005, 0.0215))))
    assert (rotation.apply(corners) + pose[:3].numpy())[:, 2].min() < -0.008
    pose[2] -= 3e-6
    assert not bool(checker.contains("part", pose, "bin"))


@pytest.mark.parametrize("batch_shape", ((), (2,), (2, 3)))
def test_containment_accepts_scalar_and_batched_poses_in_rotated_regions(tmp_path, batch_shape):
    path = tmp_path / "part.usda"
    _write_shapes(path)
    scene = SimpleNamespace(part=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=None)))
    rotation = Rotation.from_euler("xyz", [0.2, 0.3, -0.4])
    translation = np.array([0.2, -0.1, 0.3])
    region = SimpleNamespace(
        center_xyz=tuple(translation), rotation_xyzw=tuple(rotation.as_quat()), half_extents_xyz=(0.1, 0.1, 0.1)
    )
    checker = RegionContainment(scene, {"bin": region})
    pose = torch.tensor(np.r_[translation, rotation.as_quat()]).expand(*batch_shape, 7).clone()
    assert checker.contains("part", pose, "bin").shape == batch_shape
    assert bool(checker.contains("part", pose, "bin").all())
    pose[..., :3] += torch.tensor(rotation.apply([0.15, 0, 0]))
    assert not bool(checker.contains("part", pose, "bin").any())
    with pytest.raises(AssertionError, match="another dtype or device"):
        checker.contains("part", pose.float(), "bin")


@pytest.mark.parametrize("face", range(6))
def test_each_region_face_rejects_real_physical_protrusion(tmp_path, face):
    path = tmp_path / "part.usda"
    _write_shapes(path)
    scene = SimpleNamespace(part=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=None)))
    region = SimpleNamespace(center_xyz=(0, 0, 0), rotation_xyzw=(0, 0, 0, 1), half_extents_xyz=(0.1, 0.1, 0.1))
    checker = RegionContainment(scene, {"bin": region})
    pose = torch.tensor((0, 0, 0, 0, 0, 0, 1), dtype=torch.float64)
    bounds = read_collision_primitives(path, dtype=torch.float64).bounds_in_frame(pose)
    side, axis = divmod(face, 3)
    direction = -1 if side == 0 else 1
    pose[axis] = direction * 0.1 - bounds[side, axis]
    assert bool(checker.contains("part", pose, "bin"))
    pose[axis] += direction * 3e-6
    assert not bool(checker.contains("part", pose, "bin"))


def test_floor_allowance_never_expands_side_or_ceiling(tmp_path):
    path = tmp_path / "part.usda"
    _write_shapes(path)
    scene = SimpleNamespace(part=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=None)))
    region = SimpleNamespace(center_xyz=(0, 0, 0.05), rotation_xyzw=(0, 0, 0, 1), half_extents_xyz=(0.1, 0.1, 0.05))
    checker = RegionContainment(scene, {"bin": region}, floor_allowance_m=50e-6)
    pose = torch.tensor((0, 0, -0.0005 - 49e-6, 0, 0, 0, 1), dtype=torch.float64)
    assert bool(checker.contains("part", pose, "bin"))
    pose[2] -= 3e-6
    assert not bool(checker.contains("part", pose, "bin"))
    pose[2] = 0.1 - 0.0215 + 3e-6
    assert not bool(checker.contains("part", pose, "bin"))
    pose[2] = 0.02
    pose[0] = 0.1 - 0.026 + 3e-6
    assert not bool(checker.contains("part", pose, "bin"))
