# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrated RGB-D views used by CAP's syringe benchmark."""

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab_arena_environments.isaac_cap.embodiments.insertion_task.cameras import IndustrialFr3RobotiqCameraCfg

_FR3_CAMERAS = IndustrialFr3RobotiqCameraCfg()
_TOP_HALF_TILT = 0.5 * math.atan2(0.15, 0.95)
_TOP_COS = math.sqrt(0.5) * math.cos(_TOP_HALF_TILT)
_TOP_SIN = math.sqrt(0.5) * math.sin(_TOP_HALF_TILT)


@configclass
class SyringeCameraCfg(IndustrialFr3RobotiqCameraCfg):
    """CAP syringe calibration on the shared FR3 wrist and exterior camera rig."""

    top_camera: CameraCfg = _FR3_CAMERAS.top_camera.replace(
        width=1280,
        height=960,
        update_latest_camera_pose=True,
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, -0.25, 1.75),
            rot=(_TOP_COS, -_TOP_COS, -_TOP_SIN, -_TOP_SIN),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.024 / (2 * math.tan(math.radians(15))),
            focus_distance=28.0,
            horizontal_aperture=4.032,
            vertical_aperture=3.024,
            clipping_range=(0.01, 3.0),
        ),
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    """Tilted overhead RGB-D view used to locate the syringe and receiver."""

    exterior_left_camera: CameraCfg = _FR3_CAMERAS.exterior_left_camera.replace(
        width=1280,
        height=960,
        offset=CameraCfg.OffsetCfg(
            pos=(1.3213, 0.0826, 1.5159),
            # Calibrated ROS optical quaternion (x, y, z, w), looking toward
            # (-0.1495, 0.0711, 1.0465) with world Z up.
            rot=(0.5687340497970581, 0.5731982588768005, -0.4187518060207367, -0.41549041867256165),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.024 / (2 * math.tan(math.radians(22.5))),
            focus_distance=28.0,
            horizontal_aperture=4.032,
            vertical_aperture=3.024,
            clipping_range=(0.01, 5.0),
        ),
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    """Side RGB-D view used to align the held syringe with the receiver."""

    wrist_camera: CameraCfg = _FR3_CAMERAS.wrist_camera.replace(
        width=1280,
        height=720,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            distortion=sim_utils.OpenCvPinholeDistortionCfg(
                fx=1280 / (2 * math.tan(math.radians(51))),
                fy=720 / (2 * math.tan(math.radians(28.5))),
                cx=640,
                cy=360,
                image_size=(1280, 720),
                apply_lens_distortion=False,
            ),
            focal_length=5.0,
            focus_distance=28.0,
            horizontal_aperture=10 * math.tan(math.radians(51)),
            vertical_aperture=10 * math.tan(math.radians(28.5)),
        ),
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    """RGB-D view attached to the shared FR3 wrist, with live pose updates."""

    exterior_right_camera: CameraCfg = _FR3_CAMERAS.exterior_right_camera.replace(
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    """Shared exterior view retained for observation and recording."""
