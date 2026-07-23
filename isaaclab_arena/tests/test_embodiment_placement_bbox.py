# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_KITCHEN_STAND_HEIGHT_M = 0.8


def _test_droid_placement_bbox_uses_stand(simulation_app) -> bool:
    """Check Droid placement bounds come from the stand footprint, not the robot mesh."""

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

    try:
        embodiment = DroidAbsoluteJointPositionEmbodiment(stand_height_m=_KITCHEN_STAND_HEIGHT_M)
        bbox = embodiment.get_bounding_box()

        stand = embodiment.scene_config.stand
        stand_bbox = compute_local_bounding_box_from_usd(stand.spawn.usd_path, tuple(stand.spawn.scale))
        stand_offset = (0.0, 0.0, embodiment._robot_base_z_offset)
        expected = stand_bbox.translated(stand_offset)

        assert bbox.min_point.shape == expected.min_point.shape
        assert torch.allclose(bbox.min_point, expected.min_point)
        assert torch.allclose(bbox.max_point, expected.max_point)

        robot_bbox = compute_local_bounding_box_from_usd(
            embodiment.scene_config.robot.spawn.usd_path,
            tuple(embodiment.scene_config.robot.spawn.scale or (1.0, 1.0, 1.0)),
        )
        assert bbox.size[0, 0].item() > robot_bbox.size[0, 0].item()
        assert bbox.size[0, 1].item() > robot_bbox.size[0, 1].item()
        assert bbox.min_point[0, 2].item() <= 0.0

        # Relation solver passes unlifted poses; set_initial_pose applies the stand-height offset.
        floor_top_z = 0.0
        solver_z = floor_top_z - bbox.min_point[0, 2].item()
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, solver_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        spawn_z = embodiment.initial_pose.position_xyz[2]
        natural_stand_bottom_z = spawn_z + (expected.min_point[0, 2].item() - stand_offset[2])
        assert abs(natural_stand_bottom_z - floor_top_z) < 1e-3

    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        return False

    return True


def test_droid_placement_bbox_uses_stand():
    """Pytest entry point for the Droid stand placement-bbox test."""
    result = run_simulation_app_function(_test_droid_placement_bbox_uses_stand, headless=True)
    assert result, f"Test {test_droid_placement_bbox_uses_stand.__name__} failed"


if __name__ == "__main__":
    test_droid_placement_bbox_uses_stand()
