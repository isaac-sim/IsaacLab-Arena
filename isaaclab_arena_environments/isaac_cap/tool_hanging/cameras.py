# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""AUTOLab's tool-hanging camera rig: two fixed D405 views and one D405 under each YAM wrist."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena_environments.isaac_cap.embodiments.cable_routing.cameras import BimanualYamCameraCfg

_LINK_SIX_SUFFIX = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"
# Composed from yam_bimanual_scene.xml's housing -> camera_frame -> camera, in ROS optical axes.
_WRIST_POSITION = (-0.0017, 0.079729, 0.066021)
_WRIST_ROTATION_XYZW = (0.0, -0.42304971, 0.90610647, 0.0)
_TOP_POSITION = (0.300, 0.0, 1.668)
_TOP_ROTATION_XYZW = (0.6779460923076107, -0.6779460923076107, 0.200970385691128, -0.200970385691128)
_SIDE_POSITION = (0.300, -0.12, 1.668)
_SIDE_ROTATION_XYZW = (-0.6617549373273509, 0.6913939313638249, -0.14823775387287336, 0.24915939260416003)
_VERTICAL_APERTURE = 4.8


def _d405_camera(
    prim_path: str,
    position: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    width: int,
    height: int,
    vertical_fov_deg: float,
    clipping_range: tuple[float, float],
    update_latest_camera_pose: bool = False,
) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        width=width,
        height=height,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=update_latest_camera_pose,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_VERTICAL_APERTURE / (2.0 * math.tan(math.radians(vertical_fov_deg / 2.0))),
            focus_distance=28.0,
            horizontal_aperture=6.4,
            vertical_aperture=_VERTICAL_APERTURE,
            clipping_range=clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation_xyzw, convention="ros"),
    )


def _fixed_camera(
    name: str, position: tuple[float, float, float], rotation_xyzw: tuple[float, float, float, float]
) -> CameraCfg:
    return _d405_camera(f"{{ENV_REGEX_NS}}/{name}", position, rotation_xyzw, 640, 480, 55.0, (0.01, 4.0))


def _wrist_camera(robot: str, name: str) -> CameraCfg:
    return _d405_camera(
        f"{{ENV_REGEX_NS}}/{robot}{_LINK_SIX_SUFFIX}/{name}",
        _WRIST_POSITION,
        _WRIST_ROTATION_XYZW,
        320,
        240,
        58.0,
        (0.005, 1.5),
        update_latest_camera_pose=True,
    )


@configclass
class ToolHangingYamCameraCfg(BimanualYamCameraCfg):
    """Fixed overhead and side views plus both wrist cameras, named as CAP's policy expects."""

    top_camera: CameraCfg = _fixed_camera("top_camera", _TOP_POSITION, _TOP_ROTATION_XYZW)
    side_camera: CameraCfg = _fixed_camera("side_camera", _SIDE_POSITION, _SIDE_ROTATION_XYZW)
    left_wrist_camera: CameraCfg = _wrist_camera("LeftRobot", "left_wrist_camera")
    right_wrist_camera: CameraCfg = _wrist_camera("RightRobot", "right_wrist_camera")

    def set_robot_mount_positions(self, left: tuple[float, float, float], right: tuple[float, float, float]) -> None:
        """Keep the fixed cameras where AUTOLab authored them in world coordinates."""
