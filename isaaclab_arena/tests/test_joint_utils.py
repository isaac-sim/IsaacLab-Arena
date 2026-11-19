# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.utils.joint_utils import unnormalize_joint_position


@pytest.mark.parametrize(
    ("normalized_position", "joint_min", "joint_max", "expected"),
    [
        (0.25, 0.0, 2.0, 0.5),
        (0.25, -2.0, 0.0, -0.5),
    ],
)
def test_unnormalize_joint_position_preserves_open_direction(normalized_position, joint_min, joint_max, expected):
    assert unnormalize_joint_position(normalized_position, joint_min, joint_max) == pytest.approx(expected)
