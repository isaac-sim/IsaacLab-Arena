# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrated RGB-D views used by CAP's syringe benchmark."""

import math

import isaaclab.sim as sim_utils
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab_arena_environments.isaac_cap.embodiments.insertion_task.cameras import _ros_optical_quaternion


# TODO(alexmillane) [berkley-cap-align-embodiments]: Remove these per-task custom
# embodiment configurations once the upstream repo has done it.
def configure_syringe_cameras(cameras):
    """Apply CAP's syringe lenses and extrinsics to the shared FR3 camera rig."""
    top = cameras.top_camera
    top.width, top.height = 1280, 960
    half_tilt = 0.5 * math.atan2(0.15, 0.95)
    c, s = math.sqrt(0.5) * math.cos(half_tilt), math.sqrt(0.5) * math.sin(half_tilt)
    top.offset.pos = (0.1, -0.25, 1.75)
    top.offset.rot = (c, -c, -s, -s)
    top.update_latest_camera_pose = True
    top.spawn = sim_utils.PinholeCameraCfg(
        focal_length=3.024 / (2 * math.tan(math.radians(15))),
        focus_distance=28.0,
        horizontal_aperture=4.032,
        vertical_aperture=3.024,
        clipping_range=(0.01, 3.0),
    )
    side = cameras.exterior_left_camera
    side.width, side.height = 1280, 960
    side.offset.pos = (1.3213, 0.0826, 1.5159)
    side.offset.rot = _ros_optical_quaternion(side.offset.pos, (-0.1495, 0.0711, 1.0465))
    side.spawn = sim_utils.PinholeCameraCfg(
        focal_length=3.024 / (2 * math.tan(math.radians(22.5))),
        focus_distance=28.0,
        horizontal_aperture=4.032,
        vertical_aperture=3.024,
        clipping_range=(0.01, 5.0),
    )
    wrist = cameras.wrist_camera
    wrist.width, wrist.height = 1280, 720
    wrist.update_latest_camera_pose = True
    wrist.spawn = sim_utils.PinholeCameraCfg(
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
    )
    for camera in (top, side, wrist, cameras.exterior_right_camera):
        camera.data_types = ["rgb", "distance_to_image_plane"]
        camera.renderer_cfg = IsaacRtxRendererCfg()
