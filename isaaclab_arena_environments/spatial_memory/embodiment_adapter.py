# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-specific facts the suite needs: contact bodies, tool frame, egocentric camera and gripper state."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field

from isaaclab_arena_environments.spatial_memory.geometry import compose_poses
from isaaclab_arena_environments.spatial_memory.layout_utils import FLOOR_Z_M
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

    floor_z_m: float = FLOOR_Z_M
    """Floor height in the environment frame (the robot base is at the origin)."""

    registry_name: str = "franka_joint_pos"
    arm_links: tuple[str, ...] = tuple(f"panda_link{index}" for index in range(1, 8))
    gripper_bodies: tuple[str, ...] = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    tool_body: str = "panda_hand"
    body_prim_paths: dict[str, str] = field(default_factory=dict)
    """Prim path below the robot prim of each contact body whose prim is not a direct child (default: its name)."""
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

    def configure_env_cfg(self, env_cfg):
        """Return the compiled environment configuration with the embodiment's requirements applied."""
        return env_cfg

    def make_scripted_controller(self, env):
        """Return the controller that follows the suite's waypoint plans for this embodiment."""
        from isaaclab_arena_environments.spatial_memory.controller import FrankaPlanFollower

        return FrankaPlanFollower(env, self)

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


DROID_TOOL_OFFSET = (0.131, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
"""Grasp point between the Robotiq finger pads in the gripper's ``base_link`` frame (approach axis +x)."""

DROID_IMAGE_SIZE = (360, 640)
"""(height, width) of the DROID wrist image; enough for policies that resize to 224 x 224."""


@dataclass
class DroidAdapter:
    """DROID (Franka with a Robotiq 2F-85) with absolute joint-position actions and its wrist camera.

    For evaluating policies trained on DROID data on the Franka tasks; the suite's scripted controllers drive the
    Franka hand and do not support it.
    """

    floor_z_m: float = FLOOR_Z_M
    """Floor height in the environment frame (the robot base is at the origin)."""

    registry_name: str = "droid_abs_joint_pos"
    arm_links: tuple[str, ...] = tuple(f"panda_link{index}" for index in range(1, 8))
    gripper_bodies: tuple[str, ...] = (
        "robotiq_base_link",
        "robotiq_left_inner_finger",
        "robotiq_right_inner_finger",
    )
    tool_body: str = "base_link"
    body_prim_paths: dict[str, str] = field(
        default_factory=lambda: {
            "robotiq_base_link": "Gripper/Robotiq_2F_85/base_link",
            "robotiq_left_inner_finger": "Gripper/Robotiq_2F_85/left_inner_finger",
            "robotiq_right_inner_finger": "Gripper/Robotiq_2F_85/right_inner_finger",
        }
    )
    """Prim path below the robot prim of each contact body whose prim is not a direct child."""

    def make_embodiment(self, enable_cameras: bool):
        """Create the DROID embodiment on a stand that reaches the floor, rendering only a downsized wrist camera."""
        from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

        embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=enable_cameras, stand_height_m=-self.floor_z_m)
        cameras = embodiment.camera_config
        cameras.wrist_camera.height, cameras.wrist_camera.width = DROID_IMAGE_SIZE
        # The suite is egocentric: the exterior cameras are kept tiny, and nothing should read them.
        for name in ("external_camera", "external_camera_2"):
            camera = getattr(cameras, name)
            camera.height, camera.width = 18, 32
        return embodiment

    def configure_env_cfg(self, env_cfg):
        """Return the compiled environment configuration unchanged."""
        return env_cfg

    def make_scripted_controller(self, env):
        """The scripted controllers drive the Franka hand only."""
        raise NotImplementedError("The spatial-memory scripted controllers support the Franka and the G1, not DROID.")

    def camera_model(self) -> CameraModel:
        """Return the wrist camera model at the rendered resolution."""
        from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

        camera_cfg = DroidCameraCfg().wrist_camera
        camera_cfg.height, camera_cfg.width = DROID_IMAGE_SIZE
        return camera_model_from_cfg(camera_cfg, mount_body="base_link")

    def camera_obs_key(self) -> str:
        """Return the observation key of the egocentric RGB image."""
        return "wrist_camera_rgb"

    def robot_occluders(self) -> list[RobotOccluder]:
        """Return no robot boxes: the gripper is ignored, which can only overstate what the camera sees."""
        return []

    def tool_pose_w(self, env) -> torch.Tensor:
        """Return the grasp point pose (N, 7) in the world frame."""
        robot = env.scene["robot"]
        T_W_gripper = robot.data.body_link_pose_w.torch[:, robot.data.body_names.index(self.tool_body)]
        offset = torch.tensor(DROID_TOOL_OFFSET, device=env.device).expand(env.num_envs, 7)
        return compose_poses(T_W_gripper, offset)

    def gripper_closed(self, env) -> torch.Tensor:
        """Return whether the gripper is commanded closed (N,): the binary command closes above 0.5."""
        return env.action_manager.action[:, -1] > 0.5
