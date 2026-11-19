# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Waypoint plans and a batched tool-pose tracker for the Franka's scripted spatial-memory controllers.

The tracker turns Cartesian tool waypoints into joint-position actions with damped least-squares IK plus a null-space
pull towards a rest posture, which keeps the elbow in a predictable place near fixtures.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.episode_state import SpatialMemoryEpisodeState
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic

GRIPPER_OPEN = 1.0
"""Gripper command that opens the Franka hand."""

GRIPPER_CLOSE = -1.0
"""Gripper command that closes the Franka hand."""

IK_DAMPING = 0.05
"""Damping of the least-squares IK."""

NULL_SPACE_GAIN = 0.05
"""Fraction of the rest-posture error corrected per control step in the IK null space."""

NULL_SPACE_MAX_STEP_RAD = 0.02
"""Largest null-space joint correction per control step."""

TRACKING_GAIN = 0.5
"""Fraction of the remaining tool error corrected per control step (before the speed limit)."""

POSITION_DEADBAND_M = 0.002
"""Position error the tracker leaves uncorrected, so that the integrated joint command settles."""

ROTATION_DEADBAND_RAD = 0.01
"""Orientation error the tracker leaves uncorrected."""

JOINT_LIMIT_MARGIN_RAD = 0.03
"""Distance from a joint limit at which the IK stops pushing that joint further."""

MAX_COMMAND_LEAD_RAD = 0.35
"""Largest difference between commanded and measured joint positions (limits wind-up when the arm is blocked)."""

MAX_IK_LIMIT_PASSES = 3
"""Times the IK is re-solved after freezing the joints that would cross a limit."""


@dataclass
class Waypoint:
    """One target of a scripted Franka plan, expressed in the environment frame ``E``."""

    t_E_tcp: tuple[float, float, float]
    """Target position of the tool centre point."""

    q_E_tcp: tuple[float, float, float, float]
    """Target orientation of the tool centre point (x, y, z, w)."""

    gripper: float = GRIPPER_OPEN
    """Gripper command while moving to and holding this waypoint."""

    tolerance_m: float = 0.01
    """Position error below which the waypoint counts as reached."""

    tolerance_rad: float = 0.08
    """Orientation error below which the waypoint counts as reached."""

    hold_steps: int = 0
    """Control steps to stay at the waypoint after reaching it (for example to close the gripper)."""

    max_steps: int = 150
    """Control steps after which the plan advances even if the waypoint was not reached."""

    max_speed_mps: float = 0.3
    """Tool speed limit while approaching this waypoint."""

    max_angular_speed_rps: float = 1.2
    """Tool angular speed limit while approaching this waypoint."""

    gate: str | None = None
    """Name of an episode condition (see ``SpatialMemoryTaskLogic.plan_gates``) that must hold before the plan
    advances past this waypoint."""

    align_item: str | None = None
    """When set, the target is the position the named held item should reach: the tool target is shifted by the
    item-to-tool offset measured (in x and y) when the waypoint becomes active."""

    label: str = ""
    """Free-form label used in logs."""


HOLD_FOREVER_STEPS = 100_000
"""``max_steps`` of a plan's final waypoint, which holds the robot still until the episode ends."""


@dataclass
class PlanState:
    """Progress of one environment through its plan."""

    waypoints: list[Waypoint] = field(default_factory=list)
    """The plan."""

    index: int = 0
    """Index of the active waypoint."""

    steps_at_waypoint: int = 0
    """Control steps spent on the active waypoint."""

    steps_held: int = 0
    """Control steps the active waypoint has been reached (or timed out) with its gate open."""

    t_item_tcp_xy_E: tuple[float, float] | None = None
    """Horizontal item-to-tool offset of the active ``align_item`` waypoint."""

    @property
    def done(self) -> bool:
        """Whether every waypoint has been completed."""
        return self.index >= len(self.waypoints)

    def current(self) -> Waypoint | None:
        """Return the active waypoint, or the last one when the plan is done."""
        if not self.waypoints:
            return None
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]


def quat_from_matrix_columns(x_axis: tuple, y_axis: tuple, z_axis: tuple) -> tuple[float, float, float, float]:
    """Return the (x, y, z, w) quaternion of the rotation whose columns are the given axes."""
    matrix = torch.tensor([x_axis, y_axis, z_axis], dtype=torch.float32).T
    quat = math_utils.quat_from_matrix(matrix.unsqueeze(0))[0]
    return tuple(float(v) for v in quat)


def tool_quat_pointing(approach_e: tuple[float, float, float], up_hint_e: tuple[float, float, float]) -> tuple:
    """Return a tool orientation whose z axis points along ``approach_e`` and whose x axis is closest to ``up_hint_e``.

    For the Franka hand, the wrist camera sits on the hand's +x side, so ``up_hint_e`` chooses where the camera is.
    """
    z_axis = torch.tensor(approach_e, dtype=torch.float32)
    z_axis = z_axis / z_axis.norm()
    hint = torch.tensor(up_hint_e, dtype=torch.float32)
    x_axis = hint - (hint @ z_axis) * z_axis
    x_axis = x_axis / x_axis.norm()
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    return quat_from_matrix_columns(tuple(x_axis.tolist()), tuple(y_axis.tolist()), tuple(z_axis.tolist()))


def pitched_approach(yaw_rad: float, pitch_down_rad: float) -> tuple[float, float, float]:
    """Return the unit approach direction with heading ``yaw_rad`` tilted ``pitch_down_rad`` below horizontal."""
    return (
        math.cos(pitch_down_rad) * math.cos(yaw_rad),
        math.cos(pitch_down_rad) * math.sin(yaw_rad),
        -math.sin(pitch_down_rad),
    )


def top_down_quat(yaw_rad: float = 0.0) -> tuple[float, float, float, float]:
    """Return a tool orientation pointing straight down with the camera side (hand x) towards heading ``yaw_rad``."""
    return tool_quat_pointing((0.0, 0.0, -1.0), (math.cos(yaw_rad), math.sin(yaw_rad), 0.0))


def pitched_quat(yaw_rad: float, pitch_down_rad: float) -> tuple[float, float, float, float]:
    """Return a tool orientation along ``pitched_approach`` with the camera side (hand x) facing up."""
    return tool_quat_pointing(pitched_approach(yaw_rad, pitch_down_rad), (0.0, 0.0, 1.0))


def _damped_pseudo_inverse(jacobian: torch.Tensor, damping: float) -> torch.Tensor:
    """Return the damped least-squares pseudo-inverse (N, 7, 6) of a (N, 6, 7) Jacobian."""
    jacobian_t = jacobian.transpose(1, 2)
    regularizer = (damping**2) * torch.eye(jacobian.shape[1], device=jacobian.device).unsqueeze(0)
    return jacobian_t @ torch.linalg.inv(jacobian @ jacobian_t + regularizer)


class ToolPoseTracker:
    """Track per-environment waypoint plans with joint-position actions.

    Args:
        num_envs: Number of environments.
        device: Torch device.
        step_dt: Control period in seconds.
        rest_joint_pose: Arm posture (7,) the null-space term pulls towards.
    """

    def __init__(self, num_envs: int, device: str, step_dt: float, rest_joint_pose: tuple[float, ...]):
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.rest_joint_pose = torch.tensor(rest_joint_pose, device=device, dtype=torch.float32)
        self.plans = [PlanState() for _ in range(num_envs)]
        # Commanded joint positions. Integrating commands, rather than restarting from the measured joints each step,
        # removes the steady-state sag of a joint-position-controlled arm under gravity.
        self.commanded_joint_pos: torch.Tensor | None = None
        self._resync = torch.ones(num_envs, dtype=torch.bool, device=device)

    def set_plan(self, env_id: int, waypoints: list[Waypoint]) -> None:
        """Replace the plan of one environment and restart its joint command from the measured posture."""
        self.plans[env_id] = PlanState(waypoints=list(waypoints))
        self._resync[env_id] = True

    def compute_joint_targets(
        self,
        T_E_tcp: torch.Tensor,
        jacobian_tcp: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_limits: torch.Tensor,
        gates: dict[str, torch.Tensor],
        t_E_items: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return arm joint targets (N, 7) and gripper commands (N,) that move each tool towards its waypoint.

        Args:
            T_E_tcp: Current tool centre point poses in the environment frame (N, 7).
            jacobian_tcp: Tool centre point Jacobian (N, 6, 7) in world-aligned axes.
            joint_pos: Current arm joint positions (N, 7).
            joint_limits: Arm joint limits (N, 7, 2).
            gates: Boolean (N,) tensors keyed by gate name.
            t_E_items: Item positions (N, 3) in the environment frame keyed by name, for ``align_item``.
        """
        t_E_target, q_E_target, gripper, max_speed, max_angular_speed = self._active_targets(T_E_tcp, t_E_items)
        position_error, rotation_error = math_utils.compute_pose_error(
            T_E_tcp[:, :3], T_E_tcp[:, 3:7], t_E_target, q_E_target, rot_error_type="axis_angle"
        )
        self._advance_plans(position_error.norm(dim=-1), rotation_error.norm(dim=-1), gates)
        twist = self._tracking_twist(position_error, rotation_error, max_speed, max_angular_speed)
        if self.commanded_joint_pos is None:
            self.commanded_joint_pos = joint_pos.clone()
        self.commanded_joint_pos[self._resync] = joint_pos[self._resync]
        self._resync[:] = False
        delta_q, free_joints = self._solve_ik_within_joint_limits(jacobian_tcp, twist, joint_limits)
        delta_q = delta_q + self._null_space_posture_step(jacobian_tcp * free_joints.unsqueeze(1)) * free_joints
        joint_targets = self.commanded_joint_pos + delta_q
        joint_targets = torch.clamp(joint_targets, joint_pos - MAX_COMMAND_LEAD_RAD, joint_pos + MAX_COMMAND_LEAD_RAD)
        joint_targets = torch.clamp(joint_targets, joint_limits[..., 0], joint_limits[..., 1])
        self.commanded_joint_pos = joint_targets.clone()
        return joint_targets, gripper

    def _active_targets(
        self, T_E_tcp: torch.Tensor, t_E_items: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the active waypoints' target positions and orientations, gripper commands and speed limits."""
        t_E_target = T_E_tcp[:, :3].clone()
        q_E_target = T_E_tcp[:, 3:7].clone()
        gripper = torch.full((self.num_envs,), GRIPPER_OPEN, device=self.device)
        max_speed = torch.full((self.num_envs,), Waypoint.max_speed_mps, device=self.device)
        max_angular_speed = torch.full((self.num_envs,), Waypoint.max_angular_speed_rps, device=self.device)
        for env_id, plan in enumerate(self.plans):
            waypoint = plan.current()
            if waypoint is None:
                continue
            t_E_target[env_id] = torch.tensor(waypoint.t_E_tcp, device=self.device)
            if waypoint.align_item is not None and not plan.done:
                if plan.t_item_tcp_xy_E is None:
                    t_item_tcp_xy = T_E_tcp[env_id, :2] - t_E_items[waypoint.align_item][env_id, :2]
                    plan.t_item_tcp_xy_E = (float(t_item_tcp_xy[0]), float(t_item_tcp_xy[1]))
                t_E_target[env_id, 0] += plan.t_item_tcp_xy_E[0]
                t_E_target[env_id, 1] += plan.t_item_tcp_xy_E[1]
            q_E_target[env_id] = torch.tensor(waypoint.q_E_tcp, device=self.device)
            gripper[env_id] = waypoint.gripper
            max_speed[env_id] = waypoint.max_speed_mps
            max_angular_speed[env_id] = waypoint.max_angular_speed_rps
        return t_E_target, q_E_target, gripper, max_speed, max_angular_speed

    def _tracking_twist(
        self,
        position_error: torch.Tensor,
        rotation_error: torch.Tensor,
        max_speed: torch.Tensor,
        max_angular_speed: torch.Tensor,
    ) -> torch.Tensor:
        """Return this step's tool twist (N, 6, 1): a proportional step with a deadband, capped by the speed limits.

        The deadband lets the integrated joint command settle instead of fighting contact forces, for example while
        the gripper closes.
        """
        position_norm = position_error.norm(dim=-1)
        rotation_norm = rotation_error.norm(dim=-1)
        position_step = torch.minimum(
            TRACKING_GAIN * (position_norm - POSITION_DEADBAND_M).clamp_min(0.0), max_speed * self.step_dt
        )
        rotation_step = torch.minimum(
            TRACKING_GAIN * (rotation_norm - ROTATION_DEADBAND_RAD).clamp_min(0.0), max_angular_speed * self.step_dt
        )
        step_position = position_error * (position_step / position_norm.clamp_min(1e-9)).unsqueeze(-1)
        step_rotation = rotation_error * (rotation_step / rotation_norm.clamp_min(1e-9)).unsqueeze(-1)
        return torch.cat([step_position, step_rotation], dim=-1).unsqueeze(-1)

    def _solve_ik_within_joint_limits(
        self, jacobian_tcp: torch.Tensor, twist: torch.Tensor, joint_limits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the joint step (N, 7) for ``twist`` and the joints left free (N, 7).

        Joints that the step would push past a limit are frozen and the remaining joints re-solved, so that a
        saturated joint does not corrupt the tool motion.
        """
        lower = joint_limits[..., 0] + JOINT_LIMIT_MARGIN_RAD
        upper = joint_limits[..., 1] - JOINT_LIMIT_MARGIN_RAD
        free_joints = torch.ones_like(self.commanded_joint_pos, dtype=torch.bool)
        for _ in range(MAX_IK_LIMIT_PASSES):
            masked_jacobian = jacobian_tcp * free_joints.unsqueeze(1)
            delta_q = (_damped_pseudo_inverse(masked_jacobian, IK_DAMPING) @ twist).squeeze(-1) * free_joints
            next_q = self.commanded_joint_pos + delta_q
            saturating = ((next_q > upper) & (delta_q > 0)) | ((next_q < lower) & (delta_q < 0))
            if not (saturating & free_joints).any():
                break
            free_joints &= ~saturating
        return delta_q, free_joints

    def _null_space_posture_step(self, jacobian_tcp: torch.Tensor) -> torch.Tensor:
        """Return a joint step (N, 7) towards the rest posture that does not move the tool."""
        jacobian_t = jacobian_tcp.transpose(1, 2)
        # A nearly undamped projector keeps the posture correction out of the tool's motion.
        exact_inverse = jacobian_t @ torch.linalg.inv(
            jacobian_tcp @ jacobian_t + 1e-6 * torch.eye(6, device=self.device).unsqueeze(0)
        )
        null_projector = torch.eye(7, device=self.device).unsqueeze(0) - exact_inverse @ jacobian_tcp
        posture_error = (self.rest_joint_pose.unsqueeze(0) - self.commanded_joint_pos).unsqueeze(-1)
        null_step = NULL_SPACE_GAIN * (null_projector @ posture_error).squeeze(-1)
        return null_step.clamp(-NULL_SPACE_MAX_STEP_RAD, NULL_SPACE_MAX_STEP_RAD)

    def _advance_plans(
        self, position_norm: torch.Tensor, rotation_norm: torch.Tensor, gates: dict[str, torch.Tensor]
    ) -> None:
        """Move each plan to its next waypoint once the active one was reached (or timed out) and held."""
        position_norm = position_norm.tolist()
        rotation_norm = rotation_norm.tolist()
        gate_values = {name: value.tolist() for name, value in gates.items()}
        for env_id, plan in enumerate(self.plans):
            if plan.done:
                continue
            waypoint = plan.waypoints[plan.index]
            plan.steps_at_waypoint += 1
            reached = position_norm[env_id] < waypoint.tolerance_m and rotation_norm[env_id] < waypoint.tolerance_rad
            gate_open = waypoint.gate is None or bool(gate_values[waypoint.gate][env_id])
            timed_out = plan.steps_at_waypoint >= waypoint.max_steps
            if (reached or timed_out) and gate_open:
                plan.steps_held += 1
                if plan.steps_held > waypoint.hold_steps:
                    plan.index += 1
                    plan.steps_at_waypoint = 0
                    plan.steps_held = 0
                    plan.t_item_tcp_xy_E = None


class FrankaPlanFollower:
    """Follow per-environment waypoint plans on the Franka with joint-position actions.

    Args:
        env: The unwrapped environment.
        adapter: The Franka embodiment adapter.
    """

    def __init__(self, env, adapter):
        self._adapter = adapter
        self._tracker = ToolPoseTracker(env.num_envs, env.device, env.step_dt, adapter.rest_joint_pose)

    @property
    def plans(self) -> list[PlanState]:
        """Per-environment plan progress."""
        return self._tracker.plans

    def set_plan(self, env_id: int, plan: list[Waypoint]) -> None:
        """Start ``plan`` in environment ``env_id``."""
        self._tracker.set_plan(env_id, plan)

    def action(self, env, logic: SpatialMemoryTaskLogic, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return the environment action (N, A) that advances every plan by one control step."""
        adapter = self._adapter
        T_E_tcp = adapter.tracked_pose_w(env).clone()
        T_E_tcp[:, :3] -= env.scene.env_origins
        joint_pos, joint_limits = adapter.arm_joint_state(env)
        t_E_items = {name: env.arena_world.get_pose_e(name)[:, :3] for name in logic.item_names()}
        joint_targets, gripper = self._tracker.compute_joint_targets(
            T_E_tcp,
            adapter.tool_jacobian(env),
            joint_pos,
            joint_limits,
            logic.plan_gates(env, episode),
            t_E_items,
        )
        return adapter.make_action(env, joint_targets, gripper)
