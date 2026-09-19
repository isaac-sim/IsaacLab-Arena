# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Top and wrist cameras for the bimanual YAM USB-C insertion tasks."""

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.cameras import ArenaCameraCfg

_LINK_SIX_SUFFIX = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"


# TODO(xinjieyao, 09/17/2026): Remove this customization once upstream unifies camera configuration.


def _make_camera(
    prim_path: str,
    *,
    width: int = 320,
    height: int = 240,
    vertical_fov_deg: float = 58.0,
    clipping_range: tuple[float, float] = (0.005, 1.5),
    position: tuple[float, float, float] = (-0.0017, 0.079729, 0.066021),
    rotation_wxyz: tuple[float, float, float, float] = (0.0, -0.42304971, 0.90610647, 0.0),
    update_latest_camera_pose: bool = True,
) -> CameraCfg:
    """Create a USB-C camera with wrist calibration defaults.

    Args:
        prim_path: Camera prim path, including its parent attachment.
        width: Image width in pixels.
        height: Image height in pixels.
        vertical_fov_deg: Vertical field of view in degrees.
        clipping_range: Near and far clipping distances in meters.
        position: Camera position relative to its parent, in meters.
        rotation_wxyz: Camera orientation relative to its parent, using the ROS convention.
        update_latest_camera_pose: Whether to refresh the camera pose on each update.

    Returns:
        RGB and depth camera configuration.
    """
    return CameraCfg(
        prim_path=prim_path,
        width=width,
        height=height,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=update_latest_camera_pose,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=4.8 / (2.0 * math.tan(math.radians(vertical_fov_deg / 2.0))),
            focus_distance=28.0,
            horizontal_aperture=6.4,
            vertical_aperture=4.8,
            clipping_range=clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=rotation_wxyz,
            convention="ros",
        ),
    )


@configclass
class UsbcInsertionCameraCfg(ArenaCameraCfg):
    """One top camera and two wrist cameras calibrated for USB-C insertion."""

    left_wrist_camera: CameraCfg = _make_camera(f"{{ENV_REGEX_NS}}/LeftRobot{_LINK_SIX_SUFFIX}/wrist_support_camera")
    right_wrist_camera: CameraCfg = _make_camera(f"{{ENV_REGEX_NS}}/RightRobot{_LINK_SIX_SUFFIX}/wrist_camera")
    top_camera: CameraCfg = _make_camera(
        prim_path="{ENV_REGEX_NS}/topdown_camera",
        width=960,
        height=720,
        vertical_fov_deg=35.0,
        clipping_range=(0.01, 2.0),
        position=(0.44, 0.0, 1.2),
        rotation_wxyz=(-math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
        update_latest_camera_pose=False,
    )
