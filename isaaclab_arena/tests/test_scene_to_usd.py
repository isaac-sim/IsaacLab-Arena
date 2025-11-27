# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pathlib

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
EPS = 1e-6


def _test_scene_to_usd(simulation_app, output_path: pathlib.Path) -> bool:

    from pxr import Gf, Usd

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    # Set up a test scene
    asset_registry = AssetRegistry()

    kitchen = asset_registry.get_asset_by_name("kitchen")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    kitchen_initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    kitchen.set_initial_pose(kitchen_initial_pose)
    cracker_box_initial_pose = Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    cracker_box.set_initial_pose(cracker_box_initial_pose)

    # Composed scene
    scene = Scene(assets=[kitchen, cracker_box])

    # Save the scene to a USD file on disk
    print(f"Saving scene to {output_path}")
    scene.export_to_usd(output_path)

    # Load the USD file and check that the scene was saved correctly
    stage = Usd.Stage.Open(output_path.as_posix())
    root_prim = stage.GetDefaultPrim()
    assert root_prim.GetPath() == "/World"

    test_prim_names = [kitchen.name, cracker_box.name]
    test_prim_poses = {
        kitchen.name: kitchen_initial_pose,
        cracker_box.name: cracker_box_initial_pose,
    }

    # Function to convert a pxr.Gf.Quatf to a numpy array
    def to_numpy_q_wxyz(q_wxyz: Gf.Quatf) -> np.ndarray:
        return np.array([q_wxyz.GetReal(), *q_wxyz.GetImaginary()])

    # Loop over all the prims and check that the scene was saved correctly
    assert len(root_prim.GetChildren()) == len(test_prim_names)
    for prim in root_prim.GetChildren():
        prim_name = prim.GetName()
        assert prim_name in test_prim_names
        print(f"Checking prim: {prim_name}")
        prim_position = prim.GetAttribute("xformOp:translate").Get()
        prim_orientation = prim.GetAttribute("xformOp:orient").Get()
        assert np.linalg.norm(prim_position - test_prim_poses[prim_name].position_xyz) < EPS
        assert np.linalg.norm(to_numpy_q_wxyz(prim_orientation) - test_prim_poses[prim_name].rotation_wxyz) < EPS
        print(f"Prim {prim_name} position: {prim_position}")
        print(f"Prim {prim_name} orientation: {to_numpy_q_wxyz(prim_orientation)}")
        print(f"Prim {prim_name} expected position: {test_prim_poses[prim_name].position_xyz}")
        print(f"Prim {prim_name} expected orientation: {test_prim_poses[prim_name].rotation_wxyz}")

    return True


def test_scene_to_usd(tmp_path: pathlib.Path):
    # The passed tmp_path is a directory.
    output_path = tmp_path / "saved_kitchen_with_cracker_box_for_test.usd"
    result = run_simulation_app_function(
        _test_scene_to_usd,
        headless=HEADLESS,
        output_path=output_path,
    )
    assert result, "Test failed"
