# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical camera definitions for the industrial FR3 workcell."""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

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


_EXTERIOR_EYE = (0.1, 0.015, 1.75)
_EXTERNAL_EYE = (1.3213, 0.0826, 1.5159)
_EXTERNAL_LEFT_TARGET = (-0.1495, 0.0711, 1.0465)
_EXTERNAL_EYE_2 = (0.9, -0.75, 1.8)
_EXTERNAL_TARGET = (0.1, 0.015, 0.82)
_WRIST_EYE = (0.0, 0.12, -0.02)
_WRIST_TARGET = (0.0, 0.12, 0.25)


def _zed_pinhole(horizontal_fov: float, vertical_fov: float):
    # RTX needs explicit fx/fy to honor both FOVs instead of assuming square pixels.
    focal_length = 5.0
    return sim_utils.PinholeCameraCfg(
        distortion=sim_utils.OpenCvPinholeDistortionCfg(
            fx=CAMERA_WIDTH / (2 * math.tan(math.radians(horizontal_fov / 2))),
            fy=CAMERA_HEIGHT / (2 * math.tan(math.radians(vertical_fov / 2))),
            cx=CAMERA_WIDTH / 2,
            cy=CAMERA_HEIGHT / 2,
            image_size=(CAMERA_WIDTH, CAMERA_HEIGHT),
            apply_lens_distortion=False,
        ),
        focal_length=focal_length,
        focus_distance=28.0,
        horizontal_aperture=2 * focal_length * math.tan(math.radians(horizontal_fov / 2)),
        vertical_aperture=2 * focal_length * math.tan(math.radians(vertical_fov / 2)),
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
        data_types=["rgb", "distance_to_image_plane"],
        spawn=_zed_pinhole(110.0, 70.0),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERNAL_EYE,
            rot=_ros_optical_quaternion(_EXTERNAL_EYE, _EXTERNAL_LEFT_TARGET),
            convention="ros",
        ),
    )
    exterior_right_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_right_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=_zed_pinhole(110.0, 70.0),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERNAL_EYE_2,
            rot=_ros_optical_quaternion(_EXTERNAL_EYE_2, _EXTERNAL_TARGET),
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
        spawn=_zed_pinhole(102.0, 57.0),
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
        spawn=_zed_pinhole(110.0, 70.0),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERIOR_EYE,
            rot=(math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
            convention="ros",
        ),
    )

    def use_overhead_profile(self, profile: str) -> None:
        """Select the task's calibrated overhead view; retain the other three cameras."""
        camera = self.top_camera
        if profile == "syringe":
            width, height = 1280, 720
            position = (0.1, 0.015, 1.75)
            rotation = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=5.0,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
            )
        elif profile == "gear":
            width, height = 1280, 960
            fov_y = 50.0
            position = (0.0490017409436448, 0.01556502252117765, 1.33)
            rotation = (1.0, 0.0, 0.0, 0.0)
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=3.024 / (2 * math.tan(math.radians(fov_y / 2))),
                horizontal_aperture=3.024 * width / height,
                vertical_aperture=3.024,
                clipping_range=(0.01, 4.0),
            )
        elif profile == "tool_sorting":
            width, height = 1280, 960
            position = (0.3, 0.0, 2.5)
            rotation = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=2.1,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
                clipping_range=(0.01, 5.0),
            )
        else:
            raise ValueError(f"Unknown FR3 overhead profile: {profile!r}")
        camera.width, camera.height = width, height
        camera.offset.pos, camera.offset.rot = position, rotation
        # Replacing the spawn cfg also clears the shared ZED intrinsics override.
        camera.spawn = spawn
