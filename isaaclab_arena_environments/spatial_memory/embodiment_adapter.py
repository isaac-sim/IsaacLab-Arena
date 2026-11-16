# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-specific facts the suite needs: contact bodies, tool frame, egocentric camera and gripper state."""

from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab_arena_environments.spatial_memory.geometry import compose_poses
from isaaclab_arena_environments.spatial_memory.visibility import CameraModel, RobotOccluder, camera_model_from_cfg

FRANKA_TOOL_OFFSET = (0.0, 0.0, 0.107, 0.0, 0.0, 0.0, 1.0)
"""Tool centre point in the ``panda_hand`` frame (between the fingertips)."""

FRANKA_READY_JOINT_POS = (0.0, -0.785, -0.1107, -1.1775, 0.0, 0.785, 0.785)
"""Arena's Franka ready posture (arm joints)."""

FRANKA_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))

FRANKA_HAND_OCCLUDERS = (
    RobotOccluder(
        body_name="panda_hand", T_body_box=(0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0), half_extents=(0.03, 0.1, 0.035)
    ),
    RobotOccluder(
        body_name="panda_leftfinger",
        T_body_box=(0.0, 0.009, 0.027, 0.0, 0.0, 0.0, 1.0),
        half_extents=(0.011, 0.01, 0.027),
    ),
    RobotOccluder(
        body_name="panda_rightfinger",
        T_body_box=(0.0, -0.009, 0.027, 0.0, 0.0, 0.0, 1.0),
        half_extents=(0.011, 0.01, 0.027),
    ),
)
"""Box model of the Franka hand and fingers, the only robot parts the wrist camera can see."""


@dataclass
class FrankaAdapter:
    """Franka with joint-position control (``franka_joint_pos``, high-gain arm) and its wrist camera."""

    initial_joint_pose: tuple[float, ...] | None = None
    """Optional spawn joint positions of the 7 arm joints; None keeps Arena's ready posture."""

    rest_joint_pose: tuple[float, ...] = FRANKA_READY_JOINT_POS
    """Posture the scripted controllers pull the arm towards in the IK null space."""

    registry_name: str = "franka_joint_pos"
    arm_links: tuple[str, ...] = tuple(f"panda_link{index}" for index in range(1, 8))
    gripper_bodies: tuple[str, ...] = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    tool_body: str = "panda_hand"
    arm_action_name: str = "arm_action"

    def make_embodiment(self, enable_cameras: bool):
        """Create the Arena embodiment with contact reporting enabled on every robot body.

        The joint-position embodiment is given the same high-gain, gravity-compensated arm as ``franka_ik`` so that
        joint targets are tracked closely.
        """
        from isaaclab_assets.robots.franka import FRANKA_PANDA_LEGACY_HIGH_PD_CFG

        from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment, _franka_robot_cfg_on_stand

        embodiment = FrankaJointPosEmbodiment(enable_cameras=enable_cameras)
        embodiment.scene_config.robot = _franka_robot_cfg_on_stand(FRANKA_PANDA_LEGACY_HIGH_PD_CFG.copy())
        if self.initial_joint_pose is not None:
            embodiment.set_initial_joint_pose(list(self.initial_joint_pose) + [0.04, 0.04])
        embodiment.spawn_cfg_addon = {"robot": {"activate_contact_sensors": True}}
        return embodiment

    def camera_model(self) -> CameraModel:
        """Return the wrist camera model."""
        from isaaclab_arena.embodiments.franka.franka import FrankaCameraCfg

        return camera_model_from_cfg(FrankaCameraCfg().wrist_cam, mount_body="panda_hand")

    def camera_obs_key(self) -> str:
        """Return the observation key of the egocentric RGB image."""
        return "wrist_cam_rgb"

    def robot_occluders(self) -> list[RobotOccluder]:
        """Return the robot boxes that can block the camera."""
        return list(FRANKA_HAND_OCCLUDERS)

    def tool_pose_w(self, env) -> torch.Tensor:
        """Return the tool centre point pose (N, 7) in the world frame."""
        robot = env.scene["robot"]
        T_W_hand = robot.data.body_link_pose_w.torch[:, robot.data.body_names.index(self.tool_body)]
        offset = torch.tensor(FRANKA_TOOL_OFFSET, device=env.device).expand(env.num_envs, 7)
        return compose_poses(T_W_hand, offset)

    def gripper_closed(self, env) -> torch.Tensor:
        """Return whether the gripper is commanded closed (N,)."""
        return env.action_manager.action[:, -1] < 0.0

    def arm_joint_ids(self, env) -> list[int]:
        """Return the articulation indices of the arm joints."""
        names = env.scene["robot"].data.joint_names
        return [names.index(name) for name in FRANKA_ARM_JOINTS]

    def arm_joint_state(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Return arm joint positions (N, 7) and their limits (N, 7, 2)."""
        robot = env.scene["robot"]
        ids = self.arm_joint_ids(env)
        return robot.data.joint_pos.torch[:, ids], robot.data.soft_joint_pos_limits.torch[:, ids]

    def tool_jacobian(self, env) -> torch.Tensor:
        """Return the tool-point Jacobian (N, 6, 7) with respect to the arm joints, in world-aligned axes."""
        robot = env.scene["robot"]
        body_index = robot.data.body_names.index(self.tool_body)
        jacobian_index = body_index - 1 if robot.is_fixed_base else body_index
        ids = self.arm_joint_ids(env)
        jacobian = robot.data.body_link_jacobian_w.torch[:, jacobian_index, :, :][:, :, ids].clone()
        T_W_hand = robot.data.body_link_pose_w.torch[:, body_index]
        offset_b = torch.tensor(FRANKA_TOOL_OFFSET[:3], device=env.device).expand(env.num_envs, 3)
        import isaaclab.utils.math as math_utils

        offset_w = math_utils.quat_apply(T_W_hand[:, 3:7], offset_b)
        jacobian[:, :3, :] += torch.linalg.cross(
            jacobian[:, 3:, :], offset_w.unsqueeze(-1).expand_as(jacobian[:, 3:, :]), dim=1
        )
        return jacobian

    def make_action(self, env, joint_targets: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """Convert arm joint targets (N, 7) and gripper commands (N,) into the environment action."""
        term = env.action_manager.get_term(self.arm_action_name)
        arm_action = (joint_targets - term._offset) / term._scale
        return torch.cat([arm_action, gripper.unsqueeze(-1)], dim=-1)
