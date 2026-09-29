# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical camera definitions for the industrial FR3 workcell."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.cameras import ArenaCameraCfg


@configclass
class IndustrialFr3RobotiqCameraCfg(ArenaCameraCfg):
    """The calibrated overhead RGB-D camera consumed by the Gear policy."""

    top_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=960,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.024 / (2 * math.tan(math.radians(25.0))),
            horizontal_aperture=4.032,
            vertical_aperture=3.024,
            clipping_range=(0.01, 4.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0490017409436448, 0.01556502252117765, 1.33),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )
