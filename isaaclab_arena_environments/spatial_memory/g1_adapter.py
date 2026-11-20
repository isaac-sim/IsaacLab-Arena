# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 humanoid support for the suite: embodiment adapter and a walking-and-carrying plan follower.

The G1 runs Arena's whole-body controller (``g1_wbc_joint``): the lower body follows a velocity command with a learned
policy, and the upper body follows joint-position targets. A load is carried by squeezing it between the hands.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from isaaclab_arena_environments.spatial_memory.visibility import CameraModel, RobotOccluder, camera_model_from_cfg

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.episode_state import SpatialMemoryEpisodeState
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic

G1_FLOOR_Z_M = -0.78
"""Floor height below the G1's pelvis (the robot root) when it stands at the environment origin."""

G1_PELVIS_HEIGHT_COMMAND_M = 0.75
"""Pelvis height commanded to the whole-body controller."""

G1_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")
"""Arm joints an ``ArmPosture`` sets on both arms; the other joints keep their defaults."""

G1_ARM_MIRROR = {"shoulder_pitch": 1.0, "shoulder_roll": -1.0, "elbow": 1.0}
"""Sign of each right-arm joint relative to the left one for a mirrored posture."""

HOLD_ROLL_THRESHOLD_RAD = 0.15
"""The hands squeeze the load when the commanded shoulder roll is below this angle (arms drawn in) ..."""

HOLD_PITCH_THRESHOLD_RAD = -0.5
"""... and the commanded shoulder pitch is below this angle (arms raised)."""


@dataclass(frozen=True)
class ArmPosture:
    """Mirrored posture of both arms; the other arm joints keep their defaults."""

    shoulder_pitch: float
    """Shoulder pitch of the left arm in radians (negative raises the arm forwards)."""

    shoulder_roll: float
    """Shoulder roll of the left arm in radians (positive spreads the arms)."""

    elbow: float
    """Elbow angle of the left arm in radians."""


ARMS_DOWN = ArmPosture(0.0, 0.0, 0.0)
"""Arms at the controller's default posture."""

ARMS_WIDE = ArmPosture(-0.8, 0.35, 0.8)
"""Hands in front of the chest, wider apart than the load."""

ARMS_SQUEEZE = ArmPosture(-0.8, -0.15, 0.8)
"""Hands pressed against the sides of the load."""

ARMS_CARRY = ArmPosture(-1.1, -0.15, 0.8)
"""As ``ARMS_SQUEEZE``, with the load raised clear of a table."""

ARMS_PLACE = ArmPosture(-0.7, -0.15, 0.8)
"""As ``ARMS_SQUEEZE``, with the load pressed onto a table so that it stays put while the hands open."""


@dataclass
class G1Waypoint:
    """A G1 waypoint: a pelvis pose to walk to and an arm posture to reach on the way."""

    target_xy_yaw_E: tuple[float, float, float]
    """Pelvis target (x, y, yaw) in the environment frame."""

    arms: ArmPosture
    """Arm posture, interpolated from the previous waypoint's posture over ``arm_time_s``."""

    label: str = ""
    """Free-form label used in logs."""

    tolerance_m: float = 0.06
    """Pelvis position error below which the waypoint counts as reached."""

    tolerance_rad: float = 0.08
    """Pelvis heading error below which the waypoint counts as reached."""

    arm_time_s: float = 1.5
    """Time over which the arms move to ``arms``."""

    hold_steps: int = 0
    """Control steps to stay at the waypoint after reaching it."""

    max_steps: int = 2_000
    """Control steps after which the plan advances even if the waypoint was not reached."""

    max_forward_mps: float = 0.4
    """Largest forward walking speed commanded while approaching this waypoint."""


@dataclass
class G1PlanState:
    """Progress of one environment through its plan."""

    waypoints: list[G1Waypoint] = field(default_factory=list)
    """The plan."""

    index: int = 0
    """Index of the active waypoint."""

    steps_at_waypoint: int = 0
    """Control steps spent on the active waypoint."""

    steps_held: int = 0
    """Consecutive control steps the active waypoint has been reached."""

    previous_arms: ArmPosture = ARMS_DOWN
    """Arm posture of the previous waypoint, where the active waypoint's arm motion starts."""

    def current(self) -> G1Waypoint | None:
        """Return the active waypoint, or None when the plan is done."""
        return self.waypoints[self.index] if self.index < len(self.waypoints) else None


class G1PlanFollower:
    """Follow per-environment ``G1Waypoint`` plans with the whole-body controller's action.

    The action holds every joint target first, then the velocity command (forward, lateral, yaw rate), the pelvis
    height and the torso orientation.

    Args:
        env: The unwrapped environment.
    """

    POSITION_GAIN = 1.2
    """Commanded speed per metre of position error."""

    HEADING_GAIN = 1.5
    """Commanded yaw rate per radian of heading error."""

    MAX_BACKWARD_MPS = 0.3
    """Largest backward walking speed."""

    MAX_LATERAL_MPS = 0.25
    """Largest sideways walking speed."""

    MAX_YAW_RATE_RAD_S = 0.35
    """Largest turning rate; faster turns swing a carried load out of the hands."""

    def __init__(self, env):
        robot = env.scene["robot"]
        self._joint_names = list(robot.data.joint_names)
        self._default_joint_pos = robot.data.default_joint_pos.torch.clone()
        self._step_dt = env.step_dt
        self.plans = [G1PlanState() for _ in range(env.num_envs)]

    def set_plan(self, env_id: int, plan: list[G1Waypoint]) -> None:
        """Start ``plan`` in environment ``env_id``."""
        self.plans[env_id] = G1PlanState(waypoints=list(plan))

    def action(self, env, logic: SpatialMemoryTaskLogic, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return the environment action (N, A) that advances every plan by one control step."""
        num_joints = len(self._joint_names)
        velocity_command_index = num_joints
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        action[:, :num_joints] = self._default_joint_pos
        action[:, velocity_command_index + 3] = G1_PELVIS_HEIGHT_COMMAND_M
        targets = [plan.current() for plan in self.plans]
        target_xy_yaw_E = torch.tensor(
            [list(waypoint.target_xy_yaw_E) if waypoint else [0.0, 0.0, 0.0] for waypoint in targets],
            device=env.device,
        )
        position_error_P, heading_error = self._pelvis_errors(env, target_xy_yaw_E)
        for env_id, (plan, waypoint) in enumerate(zip(self.plans, targets)):
            if waypoint is None:
                self._set_arms(action, env_id, plan.previous_arms)
                continue
            fraction = min(1.0, plan.steps_at_waypoint * self._step_dt / waypoint.arm_time_s)
            self._set_arms(action, env_id, self._interpolate_arms(plan.previous_arms, waypoint.arms, fraction))
            action[env_id, velocity_command_index] = (self.POSITION_GAIN * position_error_P[env_id, 0]).clamp(
                -self.MAX_BACKWARD_MPS, waypoint.max_forward_mps
            )
            action[env_id, velocity_command_index + 1] = (self.POSITION_GAIN * position_error_P[env_id, 1]).clamp(
                -self.MAX_LATERAL_MPS, self.MAX_LATERAL_MPS
            )
            action[env_id, velocity_command_index + 2] = (self.HEADING_GAIN * heading_error[env_id]).clamp(
                -self.MAX_YAW_RATE_RAD_S, self.MAX_YAW_RATE_RAD_S
            )
            at_pose = position_error_P[env_id].norm() < waypoint.tolerance_m
            at_pose &= heading_error[env_id].abs() < waypoint.tolerance_rad
            reached = fraction >= 1.0 and bool(at_pose)
            plan.steps_at_waypoint += 1
            plan.steps_held = plan.steps_held + 1 if reached else 0
            if plan.steps_held > waypoint.hold_steps or plan.steps_at_waypoint >= waypoint.max_steps:
                plan.previous_arms = waypoint.arms
                plan.index += 1
                plan.steps_at_waypoint = 0
                plan.steps_held = 0
        return action

    def _pelvis_errors(self, env, target_xy_yaw_E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the target position relative to the pelvis in its heading frame ``P`` (N, 2) and the heading error.

        Args:
            env: The unwrapped environment.
            target_xy_yaw_E: Pelvis targets (x, y, yaw) in the environment frame (N, 3).
        """
        T_W_pelvis = env.scene["robot"].data.root_pose_w.torch
        q_W_pelvis = T_W_pelvis[:, 3:7]
        yaw_W_pelvis = math_utils.euler_xyz_from_quat(q_W_pelvis)[2]
        t_pelvis_target_W = torch.zeros(env.num_envs, 3, device=env.device)
        t_pelvis_target_W[:, :2] = target_xy_yaw_E[:, :2] + env.scene.env_origins[:, :2] - T_W_pelvis[:, :2]
        t_pelvis_target_P = math_utils.quat_apply_inverse(math_utils.yaw_quat(q_W_pelvis), t_pelvis_target_W)
        heading_error = math_utils.wrap_to_pi(target_xy_yaw_E[:, 2] - yaw_W_pelvis)
        return t_pelvis_target_P[:, :2], heading_error

    @staticmethod
    def _interpolate_arms(start: ArmPosture, end: ArmPosture, fraction: float) -> ArmPosture:
        """Return the posture ``fraction`` of the way from ``start`` to ``end``."""
        values = [
            getattr(start, joint) + fraction * (getattr(end, joint) - getattr(start, joint)) for joint in G1_ARM_JOINTS
        ]
        return ArmPosture(*values)

    def _set_arms(self, action: torch.Tensor, env_id: int, arms: ArmPosture) -> None:
        """Write the joint targets of a mirrored arm posture into one environment's action."""
        for joint in G1_ARM_JOINTS:
            value = getattr(arms, joint)
            action[env_id, self._joint_names.index(f"left_{joint}_joint")] = value
            action[env_id, self._joint_names.index(f"right_{joint}_joint")] = value * G1_ARM_MIRROR[joint]


@dataclass
class G1Adapter:
    """G1 with Arena's whole-body controller (``g1_wbc_joint``) and its head camera."""

    floor_z_m: float = G1_FLOOR_Z_M
    """Floor height in the environment frame (the pelvis starts at the origin)."""

    collision_bodies: tuple[str, ...] = (
        "pelvis",
        "torso_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
    )
    """Robot bodies whose contact with any scene part counts as a collision, apart from the load. The feet may touch
    the floor, which is not a scene part."""

    hand_bodies: tuple[str, ...] = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hand_palm_link",
        "right_hand_palm_link",
    )
    """Robot bodies that hold the load."""

    body_prim_paths: dict[str, str] = field(default_factory=dict)
    """Prim path below the robot prim of each contact body whose prim is not a direct child (default: its name)."""

    def make_embodiment(self, enable_cameras: bool):
        """Create the Arena embodiment (contact reporting is enabled on the G1 by default)."""
        from isaaclab_arena.embodiments.g1.g1 import G1WBCJointEmbodiment

        return G1WBCJointEmbodiment(enable_cameras=enable_cameras)

    def configure_env_cfg(self, env_cfg):
        """Run at 50 Hz, the rate the whole-body controller's walking policy was trained at."""
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz

        return set_control_rate_50hz(env_cfg)

    def make_scripted_controller(self, env) -> G1PlanFollower:
        """Return the controller that follows the suite's walking-and-carrying plans."""
        return G1PlanFollower(env)

    def camera_model(self) -> CameraModel:
        """Return the head camera model."""
        from isaaclab_arena.embodiments.g1.g1 import G1CameraCfg

        return camera_model_from_cfg(G1CameraCfg().robot_head_cam, mount_body="head_link")

    def camera_obs_key(self) -> str:
        """Return the observation key of the egocentric RGB image."""
        return "robot_head_cam_rgb"

    def robot_occluders(self) -> list[RobotOccluder]:
        """Return no robot boxes: the arms are ignored, which can only overstate what the camera sees."""
        return []

    def tracked_pose_w(self, env) -> torch.Tensor:
        """Return the pelvis pose (N, 7) in the world frame."""
        return env.scene["robot"].data.root_pose_w.torch.clone()

    def gripper_closed(self, env) -> torch.Tensor:
        """Return whether the hands are commanded to squeeze the load (N,)."""
        names = env.scene["robot"].data.joint_names
        action = env.action_manager.action
        roll = action[:, names.index("left_shoulder_roll_joint")]
        pitch = action[:, names.index("left_shoulder_pitch_joint")]
        return (roll < HOLD_ROLL_THRESHOLD_RAD) & (pitch < HOLD_PITCH_THRESHOLD_RAD)
