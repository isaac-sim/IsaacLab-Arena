# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena embodiments for the industrial FR3 with a Robotiq 2F-85."""

from __future__ import annotations

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

from .actions import IndustrialFr3RobotiqActionsCfg, IndustrialFr3RobotiqDifferentialIKActionsCfg
from .camera_variations import RenderedCameraExtrinsicsVariation
from .cameras import IndustrialFr3RobotiqCameraCfg
from .config import (
    ARM_JOINT_NAMES,
    END_EFFECTOR_BODY_NAME,
    IndustrialFr3RobotiqEventCfg,
    IndustrialFr3RobotiqObservationsCfg,
    IndustrialFr3RobotiqSceneCfg,
)


class IndustrialFr3Robotiq2f85EmbodimentBase(EmbodimentBase):
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
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            concatenate_observation_terms,
            arm_mode,
        )
        assert self.action_config_type is not None, "An industrial FR3 action configuration is required."
        self.scene_config = IndustrialFr3RobotiqSceneCfg()
        self.action_config = self.action_config_type()
        self.observation_config = IndustrialFr3RobotiqObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms
        self.camera_config = IndustrialFr3RobotiqCameraCfg()
        self.camera_config.use_tiled_camera = False
        self.event_config = IndustrialFr3RobotiqEventCfg()
        self.reward_config = None
        self.mimic_env = None

        if initial_joint_pose is not None:
            assert len(initial_joint_pose) == len(
                ARM_JOINT_NAMES
            ), f"Expected {len(ARM_JOINT_NAMES)} FR3 arm positions, got {len(initial_joint_pose)}."
            self.scene_config.robot.init_state.joint_pos.update(
                dict(zip(ARM_JOINT_NAMES, initial_joint_pose, strict=True))
            )

        self.add_variation(RenderedCameraExtrinsicsVariation(camera_name="wrist_camera"))
        self.add_variation(RenderedCameraExtrinsicsVariation(camera_name="top_camera"))

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return END_EFFECTOR_BODY_NAME

    def get_command_body_name(self) -> str:
        return END_EFFECTOR_BODY_NAME


@register_asset
class IndustrialFr3Robotiq2f85Embodiment(IndustrialFr3Robotiq2f85EmbodimentBase):
    """Fixed-base FR3 absolute-joint embodiment with DROID-compatible streams."""

    name = "industrial_fr3_robotiq_2f85"
    tags = ["embodiment"]
    action_config_type = IndustrialFr3RobotiqActionsCfg


@register_asset
class IndustrialFr3Robotiq2f85DifferentialIKEmbodiment(IndustrialFr3Robotiq2f85EmbodimentBase):
    """FR3 relative Cartesian control for keyboard teleoperation."""

    name = "industrial_fr3_robotiq_2f85_differential_ik"
    tags = ["embodiment"]
    action_config_type = IndustrialFr3RobotiqDifferentialIKActionsCfg
