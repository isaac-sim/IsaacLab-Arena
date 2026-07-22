# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def _test_droid_placement_bbox_unions_robot_and_stand(simulation_app):
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.embodiment_placement import compute_embodiment_placement_bbox

    embodiment = DroidAbsoluteJointPositionEmbodiment(stand_height_m=0.8)
    bbox = embodiment.get_bounding_box()
    placement_sources = embodiment.get_placement_usd_sources()
    robot_only = placement_sources[0]
    robot_bbox = compute_embodiment_placement_bbox([robot_only])

    assert isinstance(bbox, AxisAlignedBoundingBox)
    assert bbox.size[0, 0] >= robot_bbox.size[0, 0]
    assert bbox.size[0, 1] >= robot_bbox.size[0, 1]
    # Stand extends below the robot origin; without it the kitchen "on floor" pose
    # places the base under the floor once stand-height lift is applied.
    assert bbox.min_point[0, 2] < robot_bbox.min_point[0, 2]
    assert placement_sources[1].offset_pose.position_xyz == (-0.05, 0.0, 0.0)
    return True


def _test_droid_accepts_generated_stand_height_alias(simulation_app):
    """Agentic graph specs using stand_height retain the stand-height behavior."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

    legacy = DroidAbsoluteJointPositionEmbodiment(stand_height=1.3)
    explicit = DroidAbsoluteJointPositionEmbodiment(stand_height_m=1.3)

    assert legacy.scene_config.stand.spawn.scale == explicit.scene_config.stand.spawn.scale
    assert legacy.scene_config.robot.init_state.pos == explicit.scene_config.robot.init_state.pos
    return True


def test_droid_placement_bbox_unions_robot_and_stand():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_droid_placement_bbox_unions_robot_and_stand)


def test_droid_accepts_generated_stand_height_alias():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_droid_accepts_generated_stand_height_alias)
