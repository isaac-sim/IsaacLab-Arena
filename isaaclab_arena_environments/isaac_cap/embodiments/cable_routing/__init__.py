# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena embodiment for the benchmark-owned bimanual I2RT YAM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

from .actions import BimanualYamActionsCfg
from .cameras import BimanualYamCameraCfg
from .config import BimanualYamSceneCfg, END_EFFECTOR_BODY_NAME, make_yam_articulation_cfg
from .observations import BimanualYamObservationsCfg


class IndustrialBimanualYamEmbodiment(EmbodimentBase):
    """Two fixed-base YAM manipulators controlled by absolute joint targets."""

    name = "industrial_bimanual_yam"
    tags: ClassVar[list[str]] = ["embodiment", "yam", "bimanual"]
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        *,
        robot_usd_path: str,
        instanceable_robot_usd_path: str,
        left_mount_position: Sequence[float],
        right_mount_position: Sequence[float],
        enable_cameras: bool = False,
        use_tiled_cameras: bool = False,
        use_instanceable_meshes: bool = False,
    ) -> None:
        """Configure the fixed cable-routing YAM pair."""
        left_position = tuple(float(value) for value in left_mount_position)
        right_position = tuple(float(value) for value in right_mount_position)
        assert len(left_position) == len(right_position) == 3, "YAM mount positions must contain three values."
        self._robot_usd_path = robot_usd_path
        self._instanceable_robot_usd_path = instanceable_robot_usd_path
        active_usd_path = instanceable_robot_usd_path if use_instanceable_meshes else robot_usd_path

        super().__init__(
            enable_cameras=enable_cameras,
            concatenate_observation_terms=True,
            arm_mode=ArmMode.DUAL_ARM,
        )
        self.scene_config = BimanualYamSceneCfg(
            left_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/LeftRobot", left_position, active_usd_path),
            right_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/RightRobot", right_position, active_usd_path),
        )
        self.action_config = BimanualYamActionsCfg()
        self.observation_config = BimanualYamObservationsCfg()
        self.camera_config = BimanualYamCameraCfg() if enable_cameras else None
        if self.camera_config is not None:
            self.camera_config.set_use_tiled_camera(use_tiled_cameras)
            self.camera_config.set_robot_mount_positions(left_position, right_position)
            self.add_camera_variations(self.camera_config)

    def get_scene_key(self) -> str:
        """Return the left articulation as the primary scene key."""
        return "left_robot"

    def get_initial_pose(self) -> Pose:
        """Return the midpoint pose of the fixed bimanual layout."""
        left = self.scene_config.left_robot.init_state.pos
        right = self.scene_config.right_robot.init_state.pos
        midpoint = tuple((float(a) + float(b)) * 0.5 for a, b in zip(left, right, strict=True))
        return Pose(position_xyz=midpoint)

    def set_joint_initial_pos(self, joint_pos: Mapping[str, float]) -> None:
        """Update both YAM articulations' initial joint positions."""
        self.scene_config.left_robot.init_state.joint_pos.update(joint_pos)
        self.scene_config.right_robot.init_state.joint_pos.update(joint_pos)

    def get_command_body_name(self) -> str:
        return END_EFFECTOR_BODY_NAME

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return END_EFFECTOR_BODY_NAME


__all__ = ["IndustrialBimanualYamEmbodiment"]
