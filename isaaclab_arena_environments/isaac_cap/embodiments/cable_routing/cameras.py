# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cameras for the bimanual YAM cable-routing embodiment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.cameras import ArenaCameraCfg

_CAMERA_WIDTH = 1280
_CAMERA_HEIGHT = 720
_D405_MOUNT_POSITION = (-0.0107, 0.079729, 0.066021)
_D405_MOUNT_ROTATION_XYZW = (0.423, 0.0, 0.0, 0.906)
_TOP_CAMERA_OFFSET_FROM_ROBOT_MIDPOINT = (0.335, 0.0, 0.93732053)
_TOP_CAMERA_ROTATION_XYZW = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
_LINK_SIX_SUFFIX = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"


def _d405_camera(
    prim_path: str,
    *,
    position: tuple[float, float, float] = _D405_MOUNT_POSITION,
    rotation_xyzw: tuple[float, float, float, float] = _D405_MOUNT_ROTATION_XYZW,
) -> CameraCfg:
    vertical_aperture = 4.8
    vertical_fov_deg = 58.0
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
class BimanualYamCameraCfg(ArenaCameraCfg):
    """One centered top camera and one D405 below each YAM end effector."""

    left_wrist_camera: CameraCfg = _d405_camera(f"{{ENV_REGEX_NS}}/LeftRobot{_LINK_SIX_SUFFIX}/left_wrist_camera")
    right_wrist_camera: CameraCfg = _d405_camera(
        f"{{ENV_REGEX_NS}}/RightRobot{_LINK_SIX_SUFFIX}/right_wrist_camera"
    )
    top_camera: CameraCfg = _d405_camera(
        "{ENV_REGEX_NS}/top_camera",
        position=_TOP_CAMERA_OFFSET_FROM_ROBOT_MIDPOINT,
        rotation_xyzw=_TOP_CAMERA_ROTATION_XYZW,
    )

    def set_robot_mount_positions(
        self,
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> None:
        """Place the top camera at Cap's offset from the robot midpoint."""
        midpoint = tuple((float(a) + float(b)) * 0.5 for a, b in zip(left, right, strict=True))
        self.top_camera.offset.pos = tuple(
            midpoint_axis + offset_axis
            for midpoint_axis, offset_axis in zip(midpoint, _TOP_CAMERA_OFFSET_FROM_ROBOT_MIDPOINT, strict=True)
        )
