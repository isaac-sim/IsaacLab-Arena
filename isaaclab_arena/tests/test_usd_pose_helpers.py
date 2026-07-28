# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
EPS = 1e-6


def _test_get_prim_pose_in_default_prim_frame(simulation_app):
    # Import the necessary classes.

    from pxr import Usd

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.usd_pose_helpers import get_prim_pose_in_default_prim_frame

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


def _test_get_prim_pose_rejects_scaled_reference(simulation_app):
    """A referenced prim with local scale cannot be represented by Pose and OBB."""
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd_pose_helpers import get_prim_pose_in_default_prim_frame

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    reference = UsdGeom.Xform.Define(stage, "/Root/Reference")
    reference.AddScaleOp().Set(Gf.Vec3d(2.0, 2.0, 2.0))

    with pytest.raises(AssertionError, match="must have unit scale"):
        get_prim_pose_in_default_prim_frame(reference.GetPrim(), stage)
    return True


def test_get_prim_pose_in_default_prim_frame():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function(
        _test_get_prim_pose_in_default_prim_frame,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_get_prim_pose_rejects_scaled_reference():
    """Scaled referenced prims fail before their local scale can be discarded."""
    assert run_simulation_app_function(_test_get_prim_pose_rejects_scaled_reference, headless=HEADLESS)


if __name__ == "__main__":
    test_get_prim_pose_in_default_prim_frame()
