# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena embodiment for the benchmark-owned bimanual I2RT YAM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

from .actions import BimanualYamActionsCfg
from .cameras import BimanualYamCameraCfg
from .config import END_EFFECTOR_BODY_NAME, BimanualYamSceneCfg, make_yam_articulation_cfg, make_yam_ee_frame_cfg
from .observations import BimanualYamObservationsCfg


@register_asset
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
        enable_ee_frames: bool = False,
        use_tiled_cameras: bool = False,
        use_instanceable_meshes: bool = False,
        camera_config: ArenaCameraCfg | None = None,
    ) -> None:
        """Configure the fixed YAM pair with an optional task-specific camera rig.

        When cameras are enabled, copy ``camera_config`` if supplied; otherwise,
        place the default cable-routing rig relative to the robot midpoint.
        """
        from .gripper import YamGripper

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
        self.gripper = YamGripper()
        self.scene_config = BimanualYamSceneCfg(
            left_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/LeftRobot", left_position, active_usd_path),
            right_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/RightRobot", right_position, active_usd_path),
        )
        if enable_ee_frames:
            self.scene_config.left_ee_frame = make_yam_ee_frame_cfg("{ENV_REGEX_NS}/LeftRobot", "tcp")
            self.scene_config.right_ee_frame = make_yam_ee_frame_cfg("{ENV_REGEX_NS}/RightRobot", "tcp")
        self.action_config = BimanualYamActionsCfg()
        self.observation_config = BimanualYamObservationsCfg()
        self.camera_config = None
        if enable_cameras:
            if camera_config is None:
                self.camera_config = BimanualYamCameraCfg()
                self.camera_config.set_robot_mount_positions(left_position, right_position)
            else:
                self.camera_config = camera_config.copy()
        if self.camera_config is not None:
            self.camera_config.set_use_tiled_camera(use_tiled_cameras)
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

    def get_ee_frame_transformer_names(self) -> list[str]:
        """Return the enabled YAM TCP sensor names."""
        return [name for name in ("left_ee_frame", "right_ee_frame") if getattr(self.scene_config, name) is not None]

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        if arm_mode is ArmMode.DUAL_ARM:
            return END_EFFECTOR_BODY_NAME
        assert arm_mode in (ArmMode.LEFT, ArmMode.RIGHT), "A dual-arm YAM end-effector frame requires one arm side."
        return f"{arm_mode.value}_ee_frame"


__all__ = ["IndustrialBimanualYamEmbodiment"]
