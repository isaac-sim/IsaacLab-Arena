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
    size = bbox.size[0]
    return (float(size[0]), float(size[1]), float(size[2]))


def _test_compute_local_bounding_box_from_usd(simulation_app, asset_dir: pathlib.Path) -> bool:
    unit_cube = asset_dir / "unit_cube.usd"
    scaled_root_cube = asset_dir / "scaled_root_cube.usd"
    _write_cube_asset_usd(unit_cube, cube_size=1.0)
    _write_cube_asset_usd(scaled_root_cube, cube_size=1.0, root_scale=(0.5, 0.5, 0.5))

    # Spawn scale only — no in-file scale.
    size = _bbox_size(unit_cube, scale=(2.0, 2.0, 2.0))
    assert all(abs(dim - 2.0) < EPS for dim in size), size

    # Default-prim root scale is unbaked before spawn scale is applied.
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
    min_pt = bbox.min_point[0].tolist()
    max_pt = bbox.max_point[0].tolist()
    expected_min = [tx - half, ty - half, tz - half]
    expected_max = [tx + half, ty + half, tz + half]
    assert all(abs(a - b) < EPS for a, b in zip(min_pt, expected_min)), (min_pt, expected_min)
    assert all(abs(a - b) < EPS for a, b in zip(max_pt, expected_max)), (max_pt, expected_max)

    # Full default-prim bounds must be at least as large as the child-only bounds.
    full = compute_local_bounding_box_from_usd(child_usd.as_posix(), scale=spawn_scale)
    assert (full.min_point <= bbox.min_point).all()
    assert (full.max_point >= bbox.max_point).all()
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
