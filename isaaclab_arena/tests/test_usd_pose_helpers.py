# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
EPS = 1e-6


def _test_get_prim_pose_in_default_prim_frame(simulation_app):
    # Import the necessary classes.

    from pxr import Usd

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.utils.usd_pose_helpers import get_prim_pose_in_default_prim_frame

    asset_registry = AssetRegistry()
    kitchen = asset_registry.get_asset_by_name("kitchen")()

    print(f"Opening USD at: {kitchen.usd_path}")
    stage = Usd.Stage.Open(kitchen.usd_path)
    prim = stage.GetPrimAtPath("/kitchen/food_packages")

    pose = get_prim_pose_in_default_prim_frame(prim, stage)
    print(f"Position relative to default prim: {pose.position_xyz}")
    print(f"Orientation (quaternion wxyz) relative to default prim: {pose.rotation_wxyz}")

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
    result = run_simulation_app_function(
        _test_get_prim_pose_in_default_prim_frame,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_get_prim_pose_in_default_prim_frame()
