# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class _LookAtObject:
    def __init__(self, initial_pose):
        self.name = "lookat_object"
        self._initial_pose = initial_pose

    def get_initial_pose(self):
        return self._initial_pose


def test_viewer_cfg_look_at_object_handles_pose_per_env():
    from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    pose = PosePerEnv(
        poses=[
            Pose(position_xyz=(1.0, 2.0, 3.0)),
            Pose(position_xyz=(4.0, 5.0, 6.0)),
        ]
    )

    cfg = get_viewer_cfg_look_at_object(_LookAtObject(pose), offset=np.array([0.5, 0.25, 1.0]))

    assert cfg.lookat == (1.0, 2.0, 3.0)
    assert cfg.eye == (1.5, 2.25, 4.0)
    assert cfg.origin_type == "env"
