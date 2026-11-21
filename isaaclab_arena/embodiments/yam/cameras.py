# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wrist and top-down cameras for the YAM embodiment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.cameras import ArenaCameraCfg

_CAMERA_WIDTH = 1280
_CAMERA_HEIGHT = 720
# AUTOLab's yam_bimanual_scene.xml mounts the D405 housing at (-0.0107, 0.079729, 0.066021) tilted 50 degrees about X.
# Its optical frame adds 9 mm along the housing X axis and a 180 degree roll, then converts MuJoCo's (-Z forward,
# +Y up) camera axes to ROS optical axes.
_D405_MOUNT_POSITION = (-0.0017, 0.079729, 0.066021)
_D405_MOUNT_ROTATION_XYZW = (0.0, -0.42304971, 0.90610647, 0.0)
_TOP_CAMERA_OFFSET_FROM_BASE = (0.335, 0.0, 0.93732053)
_TOP_CAMERA_ROTATION_XYZW = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
_BASE_PRIM_SUFFIX = "/Geometry/arm"
_LINK_SIX_PRIM_SUFFIX = f"{_BASE_PRIM_SUFFIX}/link_1/link_2/link_3/link_4/link_5/link_6"


def make_d405_camera_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    vertical_fov_deg: float = 58.0,
    vertical_aperture: float = 4.8,
) -> CameraCfg:
    """Build a 1280x720 RealSense D405 RGB and depth camera mounted at the given offset from its parent prim.

    Args:
        prim_path: Camera prim path, under the prim it is mounted on.
        position: Camera position in its parent prim's frame, in meters.
        rotation_xyzw: Camera rotation in its parent prim's frame, in the ROS optical convention.
        vertical_fov_deg: Vertical field of view in degrees, which sets the focal length.
        vertical_aperture: Vertical film aperture in millimeters.
    """
    focal_length = vertical_aperture / (2.0 * math.tan(math.radians(vertical_fov_deg / 2.0)))
    return CameraCfg(
        prim_path=prim_path,
        height=_CAMERA_HEIGHT,
        width=_CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=28.0,
            horizontal_aperture=6.4,
            vertical_aperture=vertical_aperture,
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation_xyzw, convention="ros"),
    )


@configclass
class YamCameraCfg(ArenaCameraCfg):
    """One wrist camera on the end-effector link and one top-down camera above the robot base."""

    wrist_camera: CameraCfg = make_d405_camera_cfg(
        f"{{ENV_REGEX_NS}}/Robot{_LINK_SIX_PRIM_SUFFIX}/wrist_camera",
        _D405_MOUNT_POSITION,
        _D405_MOUNT_ROTATION_XYZW,
    )
    top_camera: CameraCfg = make_d405_camera_cfg(
        f"{{ENV_REGEX_NS}}/Robot{_BASE_PRIM_SUFFIX}/top_camera",
        _TOP_CAMERA_OFFSET_FROM_BASE,
        _TOP_CAMERA_ROTATION_XYZW,
    )
