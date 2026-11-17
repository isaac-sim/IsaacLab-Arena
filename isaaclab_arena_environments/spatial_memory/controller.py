# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Waypoint plans and a batched tool-pose tracker for the scripted spatial-memory controllers.

The tracker turns Cartesian tool waypoints into joint-position actions with damped least-squares IK plus a null-space
pull towards a rest posture, which keeps the elbow in a predictable place near fixtures.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field

import isaaclab.utils.math as math_utils

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

IK_DAMPING = 0.05
"""Damping of the least-squares IK."""

NULL_SPACE_GAIN = 0.05
"""Fraction of the rest-posture error corrected per control step in the IK null space."""

NULL_SPACE_MAX_STEP_RAD = 0.02
"""Largest null-space joint correction per control step."""

TRACKING_GAIN = 0.5
"""Fraction of the remaining tool error corrected per control step (before the speed limit)."""

POSITION_DEADBAND_M = 0.002
ROTATION_DEADBAND_RAD = 0.01

JOINT_LIMIT_MARGIN_RAD = 0.03
"""Distance from a joint limit at which the IK stops pushing that joint further."""

MAX_COMMAND_LEAD_RAD = 0.35
"""Largest difference between commanded and measured joint positions (limits wind-up when the arm is blocked)."""


@dataclass
class Waypoint:
    """One target of a scripted plan, expressed in the environment frame ``E``."""

    position_e: tuple[float, float, float]
    """Tool centre point target position."""

    quat_e: tuple[float, float, float, float]
    """Tool orientation target (x, y, z, w)."""

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
    """Name of a runtime condition that must hold before the plan advances past this waypoint."""

    align_item: str | None = None
    """When set, the target is the position the named held item should reach: the tool target is shifted by the
    tool-to-item offset measured (in x and y) when the waypoint becomes active."""

    label: str = ""
    """Free-form label used in logs."""


@dataclass
class PlanState:
    """Progress of one environment through its plan."""

    waypoints: list[Waypoint] = field(default_factory=list)
    index: int = 0
    steps_at_waypoint: int = 0
    steps_held: int = 0
    item_offset_xy: tuple[float, float] | None = None
    """Tool-to-item offset of the active ``align_item`` waypoint."""

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
    regulariser = (damping**2) * torch.eye(jacobian.shape[1], device=jacobian.device).unsqueeze(0)
    return jacobian_t @ torch.linalg.inv(jacobian @ jacobian_t + regulariser)


class ToolPoseTracker:
    """Track per-environment waypoint plans with joint-position actions.

    Args:
        num_envs: Number of environments.
        device: Torch device.
        step_dt: Control period in seconds.
        rest_joint_pos: Arm posture (7,) the null-space term pulls towards.
    """

    def __init__(self, num_envs: int, device: str, step_dt: float, rest_joint_pos: tuple[float, ...]):
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.rest_joint_pos = torch.tensor(rest_joint_pos, device=device, dtype=torch.float32)
        self.plans = [PlanState() for _ in range(num_envs)]
        # Commanded joint positions. Integrating commands, rather than restarting from the measured joints each step,
        # removes the steady-state sag of a joint-position-controlled arm under gravity.
        self.commanded_joint_pos: torch.Tensor | None = None
        self._resync = torch.ones(num_envs, dtype=torch.bool, device=device)

    def set_plan(self, env_id: int, waypoints: list[Waypoint]) -> None:
        """Replace the plan of one environment and restart its joint command from the measured posture."""
        self.plans[env_id] = PlanState(waypoints=list(waypoints))
        self._resync[env_id] = True

    def targets(
        self, T_E_tool: torch.Tensor, item_positions_e: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return target positions, orientations, gripper commands and speed limits of the active waypoints."""
        targets_pos = T_E_tool[:, :3].clone()
        targets_quat = T_E_tool[:, 3:7].clone()
        gripper = torch.full((self.num_envs,), GRIPPER_OPEN, device=self.device)
        max_speed = torch.full((self.num_envs,), 0.3, device=self.device)
        max_angular = torch.full((self.num_envs,), 1.0, device=self.device)
        for env_id, plan in enumerate(self.plans):
            waypoint = plan.current()
            if waypoint is None:
                continue
            targets_pos[env_id] = torch.tensor(waypoint.position_e, device=self.device)
            if waypoint.align_item is not None and item_positions_e is not None and not plan.done:
                if plan.item_offset_xy is None:
                    offset = T_E_tool[env_id, :2] - item_positions_e[waypoint.align_item][env_id, :2]
                    plan.item_offset_xy = (float(offset[0]), float(offset[1]))
                targets_pos[env_id, 0] += plan.item_offset_xy[0]
                targets_pos[env_id, 1] += plan.item_offset_xy[1]
            targets_quat[env_id] = torch.tensor(waypoint.quat_e, device=self.device)
            gripper[env_id] = waypoint.gripper
            max_speed[env_id] = waypoint.max_speed_mps
            max_angular[env_id] = waypoint.max_angular_speed_rps
        return targets_pos, targets_quat, gripper, max_speed, max_angular

    def compute_joint_targets(
        self,
        T_E_tool: torch.Tensor,
        jacobian_tool: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_limits: torch.Tensor,
        gates: dict[str, torch.Tensor],
        item_positions_e: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return arm joint targets (N, 7) and gripper commands (N,) that move each tool towards its waypoint.

        Args:
            T_E_tool: Current tool poses in the environment frame (N, 7).
            jacobian_tool: Tool Jacobian (N, 6, 7) in the environment (world-aligned) frame.
            joint_pos: Current arm joint positions (N, 7).
            joint_limits: Arm joint limits (N, 7, 2).
            gates: Boolean (N,) tensors keyed by gate name.
            item_positions_e: Item positions (N, 3) in the environment frame keyed by name, for ``align_item``.
        """
        targets_pos, targets_quat, gripper, max_speed, max_angular = self.targets(T_E_tool, item_positions_e)
        position_error = targets_pos - T_E_tool[:, :3]
        flip = (targets_quat * T_E_tool[:, 3:7]).sum(-1) < 0
        targets_quat = torch.where(flip.unsqueeze(-1), -targets_quat, targets_quat)
        q_error = math_utils.quat_mul(targets_quat, math_utils.quat_inv(T_E_tool[:, 3:7]))
        rotation_error = math_utils.axis_angle_from_quat(q_error)
        position_norm = position_error.norm(dim=-1)
        rotation_norm = rotation_error.norm(dim=-1)
        self._advance(position_norm, rotation_norm, gates)

        # Proportional step, capped by the waypoint's speed limits, with a small deadband so that the integrated
        # joint command settles instead of fighting contact forces (for example while the gripper closes).
        position_norm_eff = (position_norm - POSITION_DEADBAND_M).clamp_min(0.0)
        rotation_norm_eff = (rotation_norm - ROTATION_DEADBAND_RAD).clamp_min(0.0)
        position_gain = torch.minimum(
            torch.full_like(position_norm, TRACKING_GAIN) * position_norm_eff,
            max_speed * self.step_dt,
        ) / position_norm.clamp_min(1e-9)
        rotation_gain = torch.minimum(
            torch.full_like(rotation_norm, TRACKING_GAIN) * rotation_norm_eff,
            max_angular * self.step_dt,
        ) / rotation_norm.clamp_min(1e-9)
        step_position = position_error * position_gain.unsqueeze(-1)
        step_rotation = rotation_error * rotation_gain.unsqueeze(-1)
        twist = torch.cat([step_position, step_rotation], dim=-1).unsqueeze(-1)
        if self.commanded_joint_pos is None:
            self.commanded_joint_pos = joint_pos.clone()
        self.commanded_joint_pos[self._resync] = joint_pos[self._resync]
        self._resync[:] = False
        # Solve the damped least-squares IK; joints that the step would push past a limit are frozen and the
        # remaining joints re-solved, so that a saturated joint does not corrupt the tool motion.
        lower = joint_limits[..., 0] + JOINT_LIMIT_MARGIN_RAD
        upper = joint_limits[..., 1] - JOINT_LIMIT_MARGIN_RAD
        free = torch.ones_like(joint_pos, dtype=torch.bool)
        for _ in range(3):
            masked_jacobian = jacobian_tool * free.unsqueeze(1)
            delta_q = (_damped_pseudo_inverse(masked_jacobian, IK_DAMPING) @ twist).squeeze(-1) * free
            next_q = self.commanded_joint_pos + delta_q
            saturating = ((next_q > upper) & (delta_q > 0)) | ((next_q < lower) & (delta_q < 0))
            if not (saturating & free).any():
                break
            free &= ~saturating
        jacobian_tool = jacobian_tool * free.unsqueeze(1)
        jacobian_t = jacobian_tool.transpose(1, 2)
        # A nearly undamped projector keeps the posture correction out of the tool's motion.
        exact_inverse = jacobian_t @ torch.linalg.inv(
            jacobian_tool @ jacobian_t + 1e-6 * torch.eye(6, device=self.device).unsqueeze(0)
        )
        null_projector = torch.eye(7, device=self.device).unsqueeze(0) - exact_inverse @ jacobian_tool
        posture_error = (self.rest_joint_pos.unsqueeze(0) - self.commanded_joint_pos).unsqueeze(-1)
        null_step = NULL_SPACE_GAIN * (null_projector @ posture_error).squeeze(-1)
        null_step = null_step.clamp(-NULL_SPACE_MAX_STEP_RAD, NULL_SPACE_MAX_STEP_RAD) * free
        delta_q = delta_q + null_step
        joint_targets = self.commanded_joint_pos + delta_q
        joint_targets = torch.maximum(
            torch.minimum(joint_targets, joint_pos + MAX_COMMAND_LEAD_RAD), joint_pos - MAX_COMMAND_LEAD_RAD
        )
        joint_targets = torch.maximum(torch.minimum(joint_targets, joint_limits[..., 1]), joint_limits[..., 0])
        self.commanded_joint_pos = joint_targets.clone()
        return joint_targets, gripper

    def _advance(
        self, position_norm: torch.Tensor, rotation_norm: torch.Tensor, gates: dict[str, torch.Tensor]
    ) -> None:
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
                    plan.item_offset_xy = None


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

    def action(self, env, logic, runtime) -> torch.Tensor:
        """Return the environment action (N, A) that advances every plan by one control step."""
        adapter = self._adapter
        T_E_tool = logic.tool_pose_w(env).clone()
        T_E_tool[:, :3] -= env.scene.env_origins
        joint_pos, joint_limits = adapter.arm_joint_state(env)
        item_positions_e = {name: env.arena_world.get_pose_e(name)[:, :3] for name in logic.item_names()}
        joint_targets, gripper = self._tracker.compute_joint_targets(
            T_E_tool,
            adapter.tool_jacobian(env),
            joint_pos,
            joint_limits,
            logic.plan_gates(env, runtime),
            item_positions_e,
        )
        return adapter.make_action(env, joint_targets, gripper)
