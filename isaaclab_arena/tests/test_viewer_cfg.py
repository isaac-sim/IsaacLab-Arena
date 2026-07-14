# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class _StubLookatObject:
    def __init__(self, initial_pose):
        self.name = "stub_object"
        self.initial_pose = initial_pose

    def get_initial_pose(self):
        return self.initial_pose


def test_get_viewer_cfg_look_at_object_pose_per_env():
    """Relation-placed objects store PosePerEnv; viewer cfg uses env 0."""
    obj = _StubLookatObject(PosePerEnv(poses=[Pose(position_xyz=(1.0, 2.0, 3.0))]))
    viewer_cfg = get_viewer_cfg_look_at_object(obj, offset=np.array([-1.0, -1.0, 1.0]))
    assert viewer_cfg.lookat == (1.0, 2.0, 3.0)
    assert viewer_cfg.eye == (0.0, 1.0, 4.0)
