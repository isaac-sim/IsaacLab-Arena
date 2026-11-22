# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wrist and top-down cameras for the bimanual YAM embodiment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.cameras import ArenaCameraCfg

_CAMERA_WIDTH = 1280
_CAMERA_HEIGHT = 720
_HORIZONTAL_APERTURE = 6.4
"""Horizontal film aperture in millimeters."""
_SQUARE_PIXEL_VERTICAL_APERTURE = _HORIZONTAL_APERTURE * _CAMERA_HEIGHT / _CAMERA_WIDTH
"""Vertical film aperture in millimeters: 3.6 mm, the horizontal aperture scaled by the 16:9 image aspect ratio.

Isaac Lab cameras assume square pixels, so a 4:3 aperture such as 4.8 mm would not give the 1280x720 image the
requested vertical field of view. With 3.6 mm and the D405's 58-degree vertical field of view, the focal length is
3.25 mm and the horizontal field of view is about 89 degrees, close to the D405's 87 degrees.
"""
# AUTOLab's yam_bimanual_scene.xml mounts the D405 housing at (-0.0107, 0.079729, 0.066021) tilted 50 degrees about X.
# Its optical frame adds 9 mm along the housing X axis and a 180 degree roll, then converts MuJoCo's (-Z forward,
# +Y up) camera axes to ROS optical axes.
_D405_MOUNT_POSITION = (-0.0017, 0.079729, 0.066021)
_D405_MOUNT_ROTATION_XYZW = (0.0, -0.42304971, 0.90610647, 0.0)
_t_M_TOP = (0.335, 0.0, 0.93732053)
"""Top camera position in the rig frame M, which sits midway between the two bases; it looks straight down."""
_TOP_CAMERA_ROTATION_XYZW = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
BASE_PRIM_SUFFIX = "/Geometry/arm"
LINK_SIX_PRIM_SUFFIX = f"{BASE_PRIM_SUFFIX}/link_1/link_2/link_3/link_4/link_5/link_6"


def make_d405_camera_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    vertical_fov_deg: float = 58.0,
    vertical_aperture: float = _SQUARE_PIXEL_VERTICAL_APERTURE,
) -> CameraCfg:
    """Build a 1280x720 RealSense D405 RGB and depth camera mounted at the given offset from its parent prim.

    Args:
        prim_path: Camera prim path, under the prim it is mounted on.
        position: Camera position in its parent prim's frame, in meters.
        rotation_xyzw: Camera rotation in its parent prim's frame, in the ROS optical convention.
        vertical_fov_deg: Vertical field of view in degrees, which sets the focal length.
        vertical_aperture: Vertical film aperture in millimeters. The default keeps pixels square, so the image
            has the requested vertical field of view.
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
            horizontal_aperture=_HORIZONTAL_APERTURE,
            vertical_aperture=vertical_aperture,
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation_xyzw, convention="ros"),
    )


@configclass
class YamBimanualCameraCfg(ArenaCameraCfg):
    """A wrist camera on each arm's end-effector link and one top-down camera above the midpoint between the bases.

    The top camera rides on the left arm's base so it follows the rig; ``mount_top_camera_on_left_base`` sets its
    offset from that base.
    """

    left_wrist_camera: CameraCfg = make_d405_camera_cfg(
        f"{{ENV_REGEX_NS}}/LeftRobot{LINK_SIX_PRIM_SUFFIX}/left_wrist_camera",
        _D405_MOUNT_POSITION,
        _D405_MOUNT_ROTATION_XYZW,
    )
    right_wrist_camera: CameraCfg = make_d405_camera_cfg(
        f"{{ENV_REGEX_NS}}/RightRobot{LINK_SIX_PRIM_SUFFIX}/right_wrist_camera",
        _D405_MOUNT_POSITION,
        _D405_MOUNT_ROTATION_XYZW,
    )
    top_camera: CameraCfg = make_d405_camera_cfg(
        f"{{ENV_REGEX_NS}}/LeftRobot{BASE_PRIM_SUFFIX}/top_camera",
        _t_M_TOP,
        _TOP_CAMERA_ROTATION_XYZW,
    )

    def mount_top_camera_on_left_base(self, t_M_L: tuple[float, float, float]) -> None:
        """Re-express the top camera's rig-frame position as an offset from the left base it rides on.

        Args:
            t_M_L: Left base position in the rig frame M, which sits midway between the two bases.
        """
        self.top_camera.offset.pos = tuple(top - left for top, left in zip(_t_M_TOP, t_M_L, strict=True))
