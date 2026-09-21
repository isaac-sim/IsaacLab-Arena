# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena embodiment for the benchmark-owned FR3 and Robotiq 2F-85."""

from __future__ import annotations

from typing import ClassVar

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariation

from .actions import IndustrialFr3RobotiqActionsCfg, IndustrialFr3RobotiqDifferentialIKActionsCfg
from .cameras import IndustrialFr3RobotiqCameraCfg
from .config import IndustrialFr3RobotiqEventCfg, IndustrialFr3RobotiqObservationsCfg, IndustrialFr3RobotiqSceneCfg


class _IndustrialFr3Robotiq2f85Base(EmbodimentBase):
    """Shared FR3/Robotiq scene setup for each arm control mode."""

    default_arm_mode = ArmMode.SINGLE_ARM
    action_config_type = None

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        use_tiled_cameras: bool = False,
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            concatenate_observation_terms,
            arm_mode,
        )
        self.scene_config = IndustrialFr3RobotiqSceneCfg()
        self.action_config = self.action_config_type()
        self.observation_config = IndustrialFr3RobotiqObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms
        self.camera_config = IndustrialFr3RobotiqCameraCfg()
        self.set_use_tiled_cameras(use_tiled_cameras)
        self.event_config = IndustrialFr3RobotiqEventCfg()
        self.reward_config = None
        self.mimic_env = None
        if initial_joint_pose is not None:
            if len(initial_joint_pose) != len(self.action_config.arm_action.joint_names):
                raise ValueError("initial_joint_pose must contain exactly seven FR3 arm positions")
            self.scene_config.robot.init_state.joint_pos.update(
                dict(
                    zip(
                        self.action_config.arm_action.joint_names,
                        initial_joint_pose,
                        strict=True,
                    )
                )
            )
        self.add_variation(CameraExtrinsicsVariation(camera_name="wrist_camera"))
        self.add_variation(CameraExtrinsicsVariation(camera_name="top_camera"))

    def set_use_tiled_cameras(self, enabled: bool) -> None:
        """Select tiled camera sensors while preserving scalar observations."""

        if not isinstance(enabled, bool):
            raise TypeError("use_tiled_cameras must be a boolean")
        self.camera_config.use_tiled_camera = enabled

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "robotiq_base"

    def get_command_body_name(self) -> str:
        return "robotiq_base"


@register_asset
class IndustrialFr3Robotiq2f85Embodiment(_IndustrialFr3Robotiq2f85Base):
    """Fixed-base FR3 absolute-joint embodiment with DROID-compatible streams."""

    name = "industrial_fr3_robotiq_2f85"
    tags: ClassVar[list[str]] = ["embodiment"]
    action_config_type = IndustrialFr3RobotiqActionsCfg


@register_asset
class IndustrialFr3Robotiq2f85DifferentialIKEmbodiment(_IndustrialFr3Robotiq2f85Base):
    """FR3 relative Cartesian control for keyboard and SpaceMouse teleoperation."""

    name = "industrial_fr3_robotiq_2f85_differential_ik"
    tags: ClassVar[list[str]] = ["embodiment"]
    action_config_type = IndustrialFr3RobotiqDifferentialIKActionsCfg
