# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def _test_droid_placement_bbox_unions_robot_and_stand(simulation_app):
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.embodiment_placement import compute_embodiment_placement_bbox

    embodiment = DroidAbsoluteJointPositionEmbodiment()
    bbox = embodiment.get_bounding_box()
    robot_only = embodiment.get_placement_usd_sources()[0]
    robot_bbox = compute_embodiment_placement_bbox([robot_only])

    assert isinstance(bbox, AxisAlignedBoundingBox)
    assert bbox.size[0, 0] >= robot_bbox.size[0, 0]
    assert bbox.size[0, 1] >= robot_bbox.size[0, 1]
    return True


def test_droid_placement_bbox_unions_robot_and_stand():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_droid_placement_bbox_unions_robot_and_stand)
