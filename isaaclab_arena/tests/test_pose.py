# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.utils.pose import Pose

# Mark all tests in this module as unit tests (no Isaac Sim required)
pytestmark = pytest.mark.unit


def test_pose_composition():
    T_B_A = Pose(position_xyz=(1.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    T_C_B = Pose(position_xyz=(2.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))

    T_C_A = T_C_B.multiply(T_B_A)

    assert T_C_A.position_xyz == (3.0, 0.0, 0.0)
    assert T_C_A.rotation_wxyz == (1.0, 0.0, 0.0, 0.0)
