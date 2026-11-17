# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 humanoid support for the suite: embodiment facts and a walking-and-carrying plan follower.

The G1 runs Arena's whole-body controller (``g1_wbc_joint``): the lower body follows a velocity command with a learned
policy, and the upper body follows joint-position targets. A load is carried by squeezing it between the hands.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field

from isaaclab_arena_environments.spatial_memory.visibility import CameraModel, RobotOccluder, camera_model_from_cfg

G1_FLOOR_Z_M = -0.78
"""Floor height below the G1's pelvis (the robot root) when it stands at the environment origin."""

G1_BASE_HEIGHT_CMD_M = 0.75
G1_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")
G1_ARM_MIRROR = {"shoulder_pitch": 1.0, "shoulder_roll": -1.0, "elbow": 1.0}
"""Sign of each right-arm joint relative to the left one for a mirrored posture."""

HOLD_ROLL_THRESHOLD_RAD = 0.15
HOLD_PITCH_THRESHOLD_RAD = -0.5
"""The hands squeeze the load when the arms are raised (shoulder pitch below this) and drawn in (roll below that)."""


@dataclass(frozen=True)
class ArmPosture:
    """Mirrored posture of both arms; the other arm joints keep their defaults."""

    shoulder_pitch: float
    shoulder_roll: float
    elbow: float


ARMS_DOWN = ArmPosture(0.0, 0.0, 0.0)
ARMS_WIDE = ArmPosture(-0.8, 0.35, 0.8)
"""Hands in front of the chest, wider apart than the load."""
ARMS_SQUEEZE = ArmPosture(-0.8, -0.15, 0.8)
"""Hands pressed against the sides of the load."""
ARMS_CARRY = ArmPosture(-1.1, -0.15, 0.8)
"""As ``ARMS_SQUEEZE``, with the load raised clear of a table."""
ARMS_PLACE = ArmPosture(-0.7, -0.15, 0.8)
"""As ``ARMS_SQUEEZE``, with the load pressed onto a table so that it stays put while the hands open."""


@dataclass
class BaseWaypoint:
    """A G1 waypoint: a pelvis pose to walk to and an arm posture to reach on the way."""

    pose_e: tuple[float, float, float] | None
    """Pelvis target (x, y, yaw) in the environment frame; None stands still."""

    arms: ArmPosture
    """Arm posture, interpolated from the previous waypoint's posture over ``arm_time_s``."""

    label: str = ""
    tolerance_m: float = 0.06
    tolerance_rad: float = 0.08
    arm_time_s: float = 1.5
    hold_steps: int = 0
    """Steps to stay once reached before moving on."""

    max_steps: int = 2_000
    """Steps after which the waypoint is abandoned (the plan moves on)."""

    max_forward_mps: float = 0.4
    gate: str | None = None


@dataclass
class BasePlanState:
    """Progress through one environment's plan."""

    waypoints: list[BaseWaypoint] = field(default_factory=list)
    index: int = 0
    steps_in_waypoint: int = 0
    steps_reached: int = 0
    previous_arms: ArmPosture = ARMS_DOWN

    def current(self) -> BaseWaypoint | None:
        return self.waypoints[self.index] if self.index < len(self.waypoints) else None


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class G1PlanFollower:
    """Follow per-environment ``BaseWaypoint`` plans with the G1 whole-body controller's action.

    Args:
        env: The unwrapped environment.
    """

    GAIN_LINEAR = 1.2
    GAIN_YAW = 1.5
    MAX_BACKWARD_MPS = 0.3
    MAX_LATERAL_MPS = 0.25
    MAX_YAW_RATE = 0.35

    def __init__(self, env):
        robot = env.scene["robot"]
        self._joint_names = list(robot.data.joint_names)
        self._default_joint_pos = robot.data.default_joint_pos.torch.clone()
        self._step_dt = env.step_dt
        self.plans = [BasePlanState() for _ in range(env.num_envs)]

    def set_plan(self, env_id: int, plan: list[BaseWaypoint]) -> None:
        """Start ``plan`` in environment ``env_id``."""
        self.plans[env_id] = BasePlanState(waypoints=list(plan))

    def _arm_targets(self, arms: ArmPosture) -> dict[str, float]:
        targets = {}
        for joint in G1_ARM_JOINTS:
            value = getattr(arms, joint)
            targets[f"left_{joint}_joint"] = value
            targets[f"right_{joint}_joint"] = value * G1_ARM_MIRROR[joint]
        return targets

    def action(self, env, logic, runtime) -> torch.Tensor:
        """Return the environment action (N, A) that advances every plan by one control step."""
        robot = env.scene["robot"]
        num_joints = len(self._joint_names)
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        action[:, :num_joints] = self._default_joint_pos
        action[:, num_joints + 3] = G1_BASE_HEIGHT_CMD_M
        position_e = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        quat = robot.data.root_quat_w.torch
        yaw = torch.atan2(
            2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
            1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
        )
        gates = logic.plan_gates(env, runtime)
        for env_id, plan in enumerate(self.plans):
            waypoint = plan.current()
            if waypoint is None:
                arms = plan.previous_arms
            else:
                fraction = min(1.0, plan.steps_in_waypoint * self._step_dt / max(waypoint.arm_time_s, 1e-6))
                arms = ArmPosture(*(
                    getattr(plan.previous_arms, joint)
                    + fraction * (getattr(waypoint.arms, joint) - getattr(plan.previous_arms, joint))
                    for joint in G1_ARM_JOINTS
                ))
            for name, value in self._arm_targets(arms).items():
                action[env_id, self._joint_names.index(name)] = value
            if waypoint is None:
                continue
            reached = fraction >= 1.0
            if waypoint.pose_e is not None:
                target = torch.tensor(waypoint.pose_e, device=env.device)
                error_e = target[:2] - position_e[env_id]
                cos_yaw, sin_yaw = torch.cos(yaw[env_id]), torch.sin(yaw[env_id])
                error_x = cos_yaw * error_e[0] + sin_yaw * error_e[1]
                error_y = -sin_yaw * error_e[0] + cos_yaw * error_e[1]
                error_yaw = _wrap_angle(target[2] - yaw[env_id])
                action[env_id, num_joints] = (self.GAIN_LINEAR * error_x).clamp(
                    -self.MAX_BACKWARD_MPS, waypoint.max_forward_mps
                )
                action[env_id, num_joints + 1] = (self.GAIN_LINEAR * error_y).clamp(
                    -self.MAX_LATERAL_MPS, self.MAX_LATERAL_MPS
                )
                action[env_id, num_joints + 2] = (self.GAIN_YAW * error_yaw).clamp(
                    -self.MAX_YAW_RATE, self.MAX_YAW_RATE
                )
                at_pose = (error_e.norm() < waypoint.tolerance_m) & (error_yaw.abs() < waypoint.tolerance_rad)
                reached = reached and at_pose.item()
            if waypoint.gate is not None:
                reached &= bool(gates[waypoint.gate][env_id])
            plan.steps_in_waypoint += 1
            plan.steps_reached = plan.steps_reached + 1 if reached else 0
            if plan.steps_reached > waypoint.hold_steps or plan.steps_in_waypoint >= waypoint.max_steps:
                plan.previous_arms = waypoint.arms
                plan.index += 1
                plan.steps_in_waypoint = 0
                plan.steps_reached = 0
        return action


@dataclass
class G1Adapter:
    """G1 with Arena's whole-body controller (``g1_wbc_joint``) and its head camera."""

    floor_z_m: float = G1_FLOOR_Z_M
    """Floor height in the environment frame (the pelvis starts at the origin)."""

    registry_name: str = "g1_wbc_joint"
    arm_links: tuple[str, ...] = (
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
    """Bodies whose contact with scene geometry is a collision (feet on the floor are not: the floor is no part)."""

    gripper_bodies: tuple[str, ...] = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hand_palm_link",
        "right_hand_palm_link",
    )
    """Bodies that hold the load."""

    tool_body: str = "pelvis"

    def make_embodiment(self, enable_cameras: bool):
        """Create the Arena embodiment (contact reporting is enabled on the G1 by default)."""
        from isaaclab_arena.embodiments.g1.g1 import G1WBCJointEmbodiment

        return G1WBCJointEmbodiment(enable_cameras=enable_cameras)

    def configure_env_cfg(self, env_cfg):
        """Run at 50 Hz, the rate the whole-body controller's walking policy was trained at."""
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz

        return set_control_rate_50hz(env_cfg)

    def make_scripted_controller(self, env) -> G1PlanFollower:
        """Return the controller that follows the suite's base-and-arm plans."""
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

    def tool_pose_w(self, env) -> torch.Tensor:
        """Return the pelvis pose (N, 7) in the world frame; the suite's path length follows it."""
        return env.scene["robot"].data.root_pose_w.torch.clone()

    def gripper_closed(self, env) -> torch.Tensor:
        """Return whether the hands are commanded to squeeze the load (N,)."""
        names = env.scene["robot"].data.joint_names
        action = env.action_manager.action
        roll = action[:, names.index("left_shoulder_roll_joint")]
        pitch = action[:, names.index("left_shoulder_pitch_joint")]
        return (roll < HOLD_ROLL_THRESHOLD_RAD) & (pitch < HOLD_PITCH_THRESHOLD_RAD)


def heading_to(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    """Return the yaw that faces from ``from_xy`` towards ``to_xy``."""
    return math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0])
