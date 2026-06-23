# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

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
        scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "xformOp:scale")
        scale_op.Set(Gf.Vec3d(*root_scale))
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


def test_compute_local_bounding_box_from_usd(tmp_path: pathlib.Path):
    result = run_simulation_app_function(
        _test_compute_local_bounding_box_from_usd,
        headless=HEADLESS,
        asset_dir=tmp_path,
    )
    assert result, "Test failed"
