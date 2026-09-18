# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The shared bimanual YAM configured the way CAP's tool-hanging graphs expect it."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab_arena_environments.isaac_cap.embodiments.cable_routing import IndustrialBimanualYamEmbodiment
from isaaclab_arena_environments.isaac_cap.embodiments.cable_routing.config import (
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_POSITION,
    GRIPPER_JOINT_NAME,
    PASSIVE_GRIPPER_JOINT_NAME,
)

from .assets import ASSET_ROOT
from .cameras import ToolHangingYamCameraCfg

_YAM_ASSET_ROOT = f"{ASSET_ROOT}/industrial__i2rt_yam"
_HOME_ARM_JOINT_POSITIONS = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
_WIDE_GRIPPER_OPEN_POSITION = 0.0475


class ToolHangingBimanualYamEmbodiment(IndustrialBimanualYamEmbodiment):
    """Bimanual YAM with AUTOLab's home pose, gripper gains, and tool-hanging camera rig."""

    name = "tool_hanging_bimanual_yam"
    camera_config_type = ToolHangingYamCameraCfg

    def __init__(
        self,
        left_mount_position: Sequence[float],
        right_mount_position: Sequence[float],
        initial_joint_pose: Sequence[float] = _HOME_ARM_JOINT_POSITIONS,
        wide_gripper: bool = False,
        enable_cameras: bool = False,
    ) -> None:
        """
        Args:
            left_mount_position: World position of the left YAM base.
            right_mount_position: World position of the right YAM base.
            initial_joint_pose: Six arm joint positions applied to both arms at reset.
            wide_gripper: Use the 47.5 mm jaw travel of AUTOLab's upstream hand instead of the stock 37.5 mm.
            enable_cameras: Spawn the tool-hanging camera rig.
        """
        variant = "i2rt_yam_upstream_gripper" if wide_gripper else "i2rt_yam_default"
        instanceable_variant = "i2rt_yam_upstream_gripper_instanceable" if wide_gripper else "i2rt_yam_instanceable"
        super().__init__(
            robot_usd_path=f"{_YAM_ASSET_ROOT}/{variant}.usda",
            instanceable_robot_usd_path=f"{_YAM_ASSET_ROOT}/{instanceable_variant}.usda",
            left_mount_position=left_mount_position,
            right_mount_position=right_mount_position,
            enable_cameras=enable_cameras,
        )
        self.set_joint_initial_pos(dict(zip(ARM_JOINT_NAMES, initial_joint_pose, strict=True)))

        # TODO(vramasamy, 2026.09.17) [berkley-cap-align-embodiments]: Remove these per-task gripper
        # settings once the shared YAM embodiment adopts them.
        for robot in (self.scene_config.left_robot, self.scene_config.right_robot):
            gripper = robot.actuators["gripper"]
            gripper.stiffness, gripper.damping, gripper.effort_limit_sim = 200.0, 14.0, 20.0
            robot.soft_joint_pos_limit_factor = 1.0

        if wide_gripper:
            open_position = _WIDE_GRIPPER_OPEN_POSITION
            self.set_joint_initial_pos({GRIPPER_JOINT_NAME: open_position, PASSIVE_GRIPPER_JOINT_NAME: -open_position})
            for action in (self.action_config.left_gripper_action, self.action_config.right_gripper_action):
                action.offset, action.scale = open_position, GRIPPER_CLOSED_POSITION - open_position
