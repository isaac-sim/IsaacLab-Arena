# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standard H1 humanoid embodiment for IsaacLab Arena.

Provides the Unitree H1 robot hardware configuration (articulation,
actuators, contact sensors) that can be shared across different tasks.
Task-specific extensions (observation layout, cameras, commands) are
defined in task-specific modules (e.g. ``isaaclab_arena.embodiments.h1.h1_vln``).

Usage::

    from isaaclab_arena.embodiments.h1.h1 import H1SceneCfg, H1CameraCfg

    class MyH1Embodiment(EmbodimentBase):
        def __init__(self):
            self.scene_config = H1SceneCfg()
            self.camera_config = H1CameraCfg()
            ...
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets import H1_MINIMAL_CFG

from isaaclab_arena.utils.pose import Pose


# ========================================================================== #
# Scene: H1 robot articulation + contact sensors                             #
# ========================================================================== #


@configclass
class H1SceneCfg:
    """Scene configuration for the H1 humanoid.

    Uses the official ``H1_MINIMAL_CFG`` from ``isaaclab_assets`` to ensure
    joint names, init state, and actuator PD gains are consistent across
    all H1-based tasks.
    """

    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )


# ========================================================================== #
# Camera: configurable head camera with optional follow camera               #
# ========================================================================== #

_DEFAULT_H1_CAMERA_OFFSET = Pose(
    position_xyz=(0.1, 0.0, 0.5),
    rotation_wxyz=(-0.5, 0.5, -0.5, 0.5),
)

_DEFAULT_H1_FOLLOW_CAMERA_OFFSET = Pose(
    position_xyz=(-1.0, 0.0, 0.57),
    rotation_wxyz=(-0.5, 0.5, -0.5, 0.5),
)


@configclass
class H1CameraCfg:
    """Configurable camera setup for the H1 humanoid.

    Supports:
      - ``robot_head_cam``: first-person camera on pelvis (always created).
      - ``robot_follow_cam``: third-person follow camera behind the robot
        (created only when ``_enable_follow_cam=True``).

    Both camera positions can be customized via ``_camera_offset`` and
    ``_follow_camera_offset``. Use ``_is_tiled_camera=True`` for parallel
    multi-env evaluation.
    """

    robot_head_cam: CameraCfg | TiledCameraCfg = MISSING
    robot_follow_cam: CameraCfg | TiledCameraCfg | None = None

    def __post_init__(self):
        is_tiled = getattr(self, "_is_tiled_camera", False)
        cam_offset = getattr(self, "_camera_offset", _DEFAULT_H1_CAMERA_OFFSET)
        enable_follow = getattr(self, "_enable_follow_cam", False)
        follow_offset = getattr(self, "_follow_camera_offset", _DEFAULT_H1_FOLLOW_CAMERA_OFFSET)
        head_camera_data_types = getattr(self, "_head_camera_data_types", ["rgb"])
        follow_camera_data_types = getattr(self, "_follow_camera_data_types", ["rgb"])

        CameraClass = TiledCameraCfg if is_tiled else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        cam_res = getattr(self, "_camera_resolution", 512)

        # First-person head camera
        self.robot_head_cam = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis/HeadCamera",
            offset=OffsetClass(
                pos=cam_offset.position_xyz,
                rot=cam_offset.rotation_wxyz,
                convention="ros",
            ),
            update_period=0.0,
            height=cam_res,
            width=cam_res,
            data_types=list(head_camera_data_types),
            spawn=sim_utils.PinholeCameraCfg(
                horizontal_aperture=54.0,
                clipping_range=(0.1, 10.0),
            ),
        )

        # Third-person follow camera (optional)
        if enable_follow:
            self.robot_follow_cam = CameraClass(
                prim_path="{ENV_REGEX_NS}/Robot/pelvis/FollowCamera",
                offset=OffsetClass(
                    pos=follow_offset.position_xyz,
                    rot=follow_offset.rotation_wxyz,
                    convention="ros",
                ),
                update_period=0.0,
                height=cam_res,
                width=cam_res,
                data_types=list(follow_camera_data_types),
                spawn=sim_utils.PinholeCameraCfg(
                    horizontal_aperture=100.0,
                    clipping_range=(0.1, 20.0),
                ),
            )
