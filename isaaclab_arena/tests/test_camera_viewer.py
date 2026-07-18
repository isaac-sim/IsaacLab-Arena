# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class _AssetWithInitialPose:
    def __init__(self, initial_pose):
        self.name = "test_asset"
        self._initial_pose = initial_pose

    def get_initial_pose(self):
        return self._initial_pose


def test_viewer_uses_first_environment_pose_for_pose_per_env():
    asset = _AssetWithInitialPose(
        PosePerEnv(
            poses=[
                Pose(position_xyz=(1.0, 2.0, 3.0)),
                Pose(position_xyz=(4.0, 5.0, 6.0)),
            ]
        )
    )

    viewer_cfg = get_viewer_cfg_look_at_object(asset, offset=np.array([-1.0, 0.5, 2.0]))

    assert viewer_cfg.lookat == (1.0, 2.0, 3.0)
    assert viewer_cfg.eye == (0.0, 2.5, 5.0)
