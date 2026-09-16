# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
EPS = 1e-4


def _write_cube_asset_usd(
    path: pathlib.Path,
    cube_size: float,
    root_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path.as_posix())
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    cube.GetSizeAttr().Set(cube_size)
    stage.SetDefaultPrim(cube.GetPrim())
    if root_scale != (1.0, 1.0, 1.0):
        xformable = UsdGeom.Xformable(cube.GetPrim())
        scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(*root_scale))
    stage.GetRootLayer().Save()


def _write_root_with_translated_child_cube(
    path: pathlib.Path,
    cube_size: float,
    child_translate: tuple[float, float, float],
    root_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """USD with default-prim root and a translated child cube (stand-like sub-prim)."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path.as_posix())
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    if root_scale != (1.0, 1.0, 1.0):
        root.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*root_scale))

    child = UsdGeom.Xform.Define(stage, "/Root/Child")
    child.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*child_translate))
    cube = UsdGeom.Cube.Define(stage, "/Root/Child/Cube")
    cube.GetSizeAttr().Set(cube_size)
    stage.GetRootLayer().Save()


def _bbox_size(path: pathlib.Path, scale: tuple[float, float, float]) -> tuple[float, float, float]:
    from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

    bbox = compute_local_bounding_box_from_usd(path.as_posix(), scale=scale)
    size = 2.0 * bbox.half_extents[0]
    return (float(size[0]), float(size[1]), float(size[2]))


def _test_compute_local_bounding_box_from_usd(simulation_app, asset_dir: pathlib.Path) -> bool:
    unit_cube = asset_dir / "unit_cube.usd"
    scaled_root_cube = asset_dir / "scaled_root_cube.usd"
    _write_cube_asset_usd(unit_cube, cube_size=1.0)
    _write_cube_asset_usd(scaled_root_cube, cube_size=1.0, root_scale=(0.5, 0.5, 0.5))

    # Spawn scale only — no in-file scale.
    size = _bbox_size(unit_cube, scale=(2.0, 2.0, 2.0))
    assert all(abs(dim - 2.0) < EPS for dim in size), size

    # ComputeUntransformedBound excludes the authored default-prim transform before spawn scale is applied.
    size = _bbox_size(scaled_root_cube, scale=(1.0, 1.0, 1.0))
    assert all(abs(dim - 1.0) < EPS for dim in size), size

    size = _bbox_size(scaled_root_cube, scale=(2.0, 2.0, 2.0))
    assert all(abs(dim - 2.0) < EPS for dim in size), size

    # grey_bin-style asset: root scale in file + matching Object.scale at spawn.
    grey_bin_style = asset_dir / "grey_bin_style.usd"
    _write_cube_asset_usd(grey_bin_style, cube_size=1.0, root_scale=(0.007, 0.007, 0.007))
    size = _bbox_size(grey_bin_style, scale=(0.007, 0.007, 0.007))
    assert all(abs(dim - 0.007) < EPS for dim in size), size

    return True


def _test_compute_local_bounding_box_from_usd_prim_path(simulation_app, asset_dir: pathlib.Path) -> bool:
    """Optional prim_path returns that sub-prim's AABB in the default-prim frame, with scale unbaking."""
    from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

    child_usd = asset_dir / "root_with_child_cube.usd"
    translate = (0.5, -0.25, -1.0)
    spawn_scale = (2.0, 2.0, 2.0)
    _write_root_with_translated_child_cube(
        child_usd, cube_size=1.0, child_translate=translate, root_scale=(0.5, 0.5, 0.5)
    )

    # Root-scale is baked into world bounds then unbaked via composed spawn/root scale.
    # Child translate under a scaled default prim therefore ends up at spawn_scale * translate.
    bbox = compute_local_bounding_box_from_usd(child_usd.as_posix(), scale=spawn_scale, prim_path="/Root/Child")
    half = 0.5 * spawn_scale[0]
    tx, ty, tz = (c * spawn_scale[0] for c in translate)
    minimum, maximum = bbox.get_axis_aligned_bounds()
    min_pt = minimum[0].tolist()
    max_pt = maximum[0].tolist()
    expected_min = [tx - half, ty - half, tz - half]
    expected_max = [tx + half, ty + half, tz + half]
    assert all(abs(a - b) < EPS for a, b in zip(min_pt, expected_min)), (min_pt, expected_min)
    assert all(abs(a - b) < EPS for a, b in zip(max_pt, expected_max)), (max_pt, expected_max)

    # Full default-prim bounds must be at least as large as the child-only bounds.
    full = compute_local_bounding_box_from_usd(child_usd.as_posix(), scale=spawn_scale)
    full_minimum, full_maximum = full.get_axis_aligned_bounds()
    assert (full_minimum <= minimum).all()
    assert (full_maximum >= maximum).all()
    return True


def _test_mesh_exclusion_errors(simulation_app, asset_dir: pathlib.Path) -> bool:
    """Distinguish fully excluded meshes from malformed included meshes."""
    import pytest
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd_helpers import NoCollisionMeshError, extract_trimesh_from_usd

    usd_path = asset_dir / "mesh_exclusions.usda"
    stage = Usd.Stage.CreateNew(usd_path.as_posix())
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    excluded_mesh = UsdGeom.Mesh.Define(stage, "/Root/Excluded")
    excluded_mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    excluded_mesh.GetFaceVertexCountsAttr().Set([3])
    excluded_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    stage.GetRootLayer().Save()

    assert extract_trimesh_from_usd(usd_path.as_posix(), excluded_prim_paths=["/Root/Excluded"]) is None

    UsdGeom.Mesh.Define(stage, "/Root/Malformed")
    stage.GetRootLayer().Save()
    with pytest.raises(NoCollisionMeshError) as error:
        extract_trimesh_from_usd(usd_path.as_posix(), excluded_prim_paths=["/Root/Excluded"])
    assert type(error.value) is NoCollisionMeshError
    return True


def test_compute_local_bounding_box_from_usd(tmp_path: pathlib.Path):
    result = run_function_with_persistent_simulation_app(
        _test_compute_local_bounding_box_from_usd,
        headless=HEADLESS,
        asset_dir=tmp_path,
    )
    assert result, "Test failed"


def test_compute_local_bounding_box_from_usd_prim_path(tmp_path: pathlib.Path):
    result = run_function_with_persistent_simulation_app(
        _test_compute_local_bounding_box_from_usd_prim_path,
        headless=HEADLESS,
        asset_dir=tmp_path,
    )
    assert result, "Test failed"


def test_mesh_exclusion_errors(tmp_path: pathlib.Path):
    assert run_function_with_persistent_simulation_app(
        _test_mesh_exclusion_errors,
        headless=HEADLESS,
        asset_dir=tmp_path,
    )


def _test_rotated_scaled_subprim_bounds(simulation_app, asset_dir: pathlib.Path) -> bool:
    """Sub-prim OBBs enclose authored scale and spawn scale in the correct frames."""
    import numpy as np
    import torch

    from isaaclab.utils.math import quat_apply_inverse
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

    path = asset_dir / "rotated_scaled.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(100, -30, 17))
    root.AddRotateXYZOp().Set(Gf.Vec3f(10, 20, 30))
    root.AddScaleOp().Set(Gf.Vec3f(3, 2, 4))
    child = UsdGeom.Xform.Define(stage, "/Root/Child")
    child.AddTranslateOp().Set(Gf.Vec3d(0.5, -0.25, 1))
    child.AddRotateXYZOp().Set(Gf.Vec3f(25, 35, 45))
    child.AddScaleOp().Set(Gf.Vec3f(2, 0.5, 3))
    cube = UsdGeom.Cube.Define(stage, "/Root/Child/Cube")
    cube.GetSizeAttr().Set(1.0)
    stage.GetRootLayer().Save()

    spawn_scale = (2.0, 3.0, 0.7)
    box = compute_local_bounding_box_from_usd(str(path), scale=spawn_scale, prim_path="/Root/Child")
    # Independent expected vertices from the authored child matrix, excluding the root transform.
    child_matrix = child.GetLocalTransformation()
    vertices = [
        np.asarray(child_matrix.Transform(Gf.Vec3d(x, y, z))) * spawn_scale
        for x in (-0.5, 0.5)
        for y in (-0.5, 0.5)
        for z in (-0.5, 0.5)
    ]
    vertices = torch.tensor(np.asarray(vertices), dtype=torch.float32)
    points_B = quat_apply_inverse(box.rotation_xyzw.expand(8, 4), vertices - box.center)
    assert (points_B.abs() <= box.half_extents + 1e-5).all()
    assert not box.is_axis_aligned().item()
    # Every local bound is tight even when the affine transform induces shear.
    torch.testing.assert_close(points_B.amin(dim=0), -box.half_extents[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(points_B.amax(dim=0), box.half_extents[0], atol=1e-5, rtol=0)
    return True


def test_rotated_scaled_subprim_bounds(tmp_path: pathlib.Path):
    assert run_function_with_persistent_simulation_app(
        _test_rotated_scaled_subprim_bounds, headless=HEADLESS, asset_dir=tmp_path
    )


def _test_mixed_unsupported_geometry_is_rejected(simulation_app):
    import pytest
    from pxr import Usd, UsdGeom

    from isaaclab_arena.utils.usd_helpers import UnsupportedCollisionGeometryError, extract_trimesh_from_prim

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    with pytest.raises(UnsupportedCollisionGeometryError, match="/Root/Cube"):
        extract_trimesh_from_prim(stage, "/Root")
    return True


def test_mixed_unsupported_geometry_is_rejected():
    assert run_function_with_persistent_simulation_app(_test_mixed_unsupported_geometry_is_rejected, headless=HEADLESS)
