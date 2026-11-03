# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
EPS = 1e-6


def _test_get_prim_pose_in_default_prim_frame(simulation_app):
    # Import the necessary classes.

    from pxr import Usd

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.usd.pose import get_prim_pose_in_default_prim_frame

    asset_registry = AssetRegistry()
    kitchen = asset_registry.get_asset_by_name("kitchen")()

    print(f"Opening USD at: {kitchen.usd_path}")
    stage = Usd.Stage.Open(kitchen.usd_path)
    prim = stage.GetPrimAtPath("/kitchen/food_packages")

    pose = get_prim_pose_in_default_prim_frame(prim, stage)
    print(f"Position relative to default prim: {pose.position_xyz}")
    print(f"Orientation (quaternion xyzw) relative to default prim: {pose.rotation_xyzw}")

    # This number is read out of the GUI from the test scene.
    pos_np_gt = np.array((2.899114282976978, -0.3971232408755399, 1.0062618326241144))

    # Here we compare the result with the number read out from the GUI.
    pos_np = np.array(pose.position_xyz)
    pos_np_diff = pos_np - pos_np_gt
    print(f"Position difference: {pos_np_diff}")

    assert np.all(pos_np_diff < EPS), "Position difference is too large"

    # NOTE(alexmillane): I haven't checked the rotation because the GUI gives
    # it in euler angles.

    return True


def test_get_prim_pose_in_default_prim_frame():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_function_with_persistent_simulation_app(
        _test_get_prim_pose_in_default_prim_frame,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def _test_get_prim_scale_in_spawned_asset(simulation_app):
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.pose import get_prim_scale_in_spawned_asset

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Asset")
    # Isaac Lab replaces the default prim's own transform, so its scale does not count.
    root.AddScaleOp().Set(Gf.Vec3f(10.0))
    stage.SetDefaultPrim(root.GetPrim())
    link = UsdGeom.Xform.Define(stage, "/Asset/Link")
    link.AddRotateZOp().Set(90.0)
    link.AddScaleOp().Set(Gf.Vec3f(3.0, 1.0, 1.0))

    # The link's x axis lies along the asset's y axis, so the spawn scale's y component stretches it.
    scale = get_prim_scale_in_spawned_asset(link.GetPrim(), (1.0, 2.0, 1.0))
    np.testing.assert_allclose(scale, [6.0, 1.0, 1.0], atol=EPS)
    return True


def test_get_prim_scale_in_spawned_asset():
    result = run_function_with_persistent_simulation_app(
        _test_get_prim_scale_in_spawned_asset,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_get_prim_pose_in_default_prim_frame()
