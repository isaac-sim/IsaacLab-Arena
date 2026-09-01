# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical camera definitions for the industrial FR3 workcell."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab_arena.utils.cameras import ArenaCameraCfg

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


def _ros_optical_quaternion(eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Return a ROS optical camera orientation aimed from ``eye`` to ``target``."""

    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    eye_tensor = torch.tensor([eye], dtype=torch.float32)
    target_tensor = torch.tensor([target], dtype=torch.float32)
    opengl_to_ros = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    rotation = create_rotation_matrix_from_view(eye_tensor, target_tensor, "Z")[0] @ opengl_to_ros
    return tuple(quat_from_matrix(rotation).tolist())


_WORKSPACE_TARGET = (0.1, 0.015, 0.82)
_TOP_CAMERA_EYE = (1.25, 0.0, 1.75)
_EXTERIOR_LEFT_EYE = (0.9, 0.75, 1.8)
_EXTERIOR_RIGHT_EYE = (0.9, -0.75, 1.8)
_WRIST_EYE = (0.0, 0.12, -0.02)
_WRIST_TARGET = (0.0, 0.12, 0.25)
_EXTERIOR_PINHOLE = dict(
    focal_length=2.1,
    focus_distance=28.0,
    horizontal_aperture=5.376,
    vertical_aperture=3.024,
)
_WRIST_PRIM = (
    "{ENV_REGEX_NS}/Robot/Geometry/base/fr3_link0/fr3_link1/"
    "fr3_link2/fr3_link3/fr3_link4/fr3_link5/fr3_link6/"
    "fr3_link7/robotiq_attach/Geometry/robotiq_base/wrist_camera"
)


@configclass
class IndustrialFr3RobotiqCameraCfg(ArenaCameraCfg):
    """One wrist, two exterior, and one top camera for the FR3 workcell."""

    exterior_left_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_left_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_EXTERIOR_PINHOLE),
        renderer_cfg=IsaacRtxRendererCfg(),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERIOR_LEFT_EYE,
            rot=_ros_optical_quaternion(_EXTERIOR_LEFT_EYE, _WORKSPACE_TARGET),
            convention="ros",
        ),
    )
    exterior_right_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_right_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_EXTERIOR_PINHOLE),
        renderer_cfg=IsaacRtxRendererCfg(),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERIOR_RIGHT_EYE,
            rot=_ros_optical_quaternion(_EXTERIOR_RIGHT_EYE, _WORKSPACE_TARGET),
            convention="ros",
        ),
    )
    wrist_camera: CameraCfg = CameraCfg(
        prim_path=_WRIST_PRIM,
        # The prim rides the wrist; without this the reported pose stays at
        # its spawn value, so anything projecting depth through it lands the
        # result where the hand was at reset rather than where it is now.
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_WRIST_EYE,
            rot=_ros_optical_quaternion(_WRIST_EYE, _WRIST_TARGET),
            convention="ros",
        ),
    )
    top_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_TOP_CAMERA_EYE,
            rot=_ros_optical_quaternion(_TOP_CAMERA_EYE, _WORKSPACE_TARGET),
            convention="ros",
        ),
    )
