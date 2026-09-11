# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pinch and drag the medium cable, demonstrate task success, and show its automatic reset."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena_environments.isaac_cap.tools import EnvBehaviourDemo

if TYPE_CHECKING:
    import torch

_MAX_ARM_JOINT_STEP = 0.04
_GRIPPER_OPEN = 0.0
_GRIPPER_CLOSED = 1.0

_HOME = (0.0, 0.85, 0.6, 0.0, 0.0, 0.0)
_RIGHT_APPROACH = (0.409067, 2.066704, 0.474773, 1.433090, 0.400142, 0.105784)

_LEFT_CABLE_SEGMENT = 82
_RIGHT_CABLE_SEGMENT = 9
_HIGH_HOVER_CLEARANCE = 0.075
_LOW_HOVER_CLEARANCE = 0.045
# The lower-finger body midpoint sits above the surfaces that contact the cable.
_GRASP_HEIGHT_OFFSET = 0.035
_DRAG_HEIGHT_OFFSET = 0.035
# The original demo continued around the peg at a 55 mm radius. Stop at a 75 mm
# radius instead so the moving gripper never enters the peg-crossing part of that path.
_DRAG_RADIUS = 0.075
_DRAG_APPROACH_STEPS = 4
_GRIPPER_PEG_CLEARANCE = 0.070
_RETREAT_CLEARANCE = 0.070

_SUCCESS_ROUTE = ((0, -1), (1, 1))
_SUCCESS_ROUTE_WINDING = 4.0
_SUCCESS_ROUTE_WRAP_RADIUS = 0.027
_SUCCESS_ROUTE_START = (-0.095, -0.145)
_SUCCESS_ROUTE_ENTRY_ANGLE = -2.10
_SUCCESS_ROUTE_ARC_SAMPLES = 20
_SUCCESS_ROUTE_MINIMUM_BEND_RADIUS = 0.012
_SUCCESS_ROUTE_BOARD_HALF_SIZE = (0.15, 0.20)
_SUCCESS_ROUTE_BOARD_MARGIN = 0.025


def _mirror_yam_joints(joints: tuple[float, ...]) -> tuple[float, ...]:
    """Mirror a right-YAM cable pose onto the left YAM."""
    q1, q2, q3, q4, q5, q6 = joints
    return (-q1, q2, q3, q4, -q5, -q6)


def _quat_mul_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply batches of XYZW quaternions."""
    import torch

    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _sample_exact_length_curve(
    controls: torch.Tensor,
    num_segments: int,
    rest_length: float,
) -> torch.Tensor:
    """Sample a guide into an exact-length, bend-limited planar cable curve."""
    import torch

    edge = controls[1:] - controls[:-1]
    edge_length = torch.linalg.vector_norm(edge, dim=-1)
    cumulative_length = torch.cat((torch.zeros_like(edge_length[:1]), edge_length.cumsum(dim=0)))
    required_length = num_segments * rest_length
    assert cumulative_length[-1] >= required_length, "Success-route guide is shorter than the cable."

    distance = torch.arange(num_segments + 1, device=controls.device, dtype=controls.dtype) * rest_length
    segment = torch.searchsorted(cumulative_length.contiguous(), distance.contiguous(), right=True) - 1
    segment = segment.clamp(min=0, max=controls.shape[0] - 2)
    start_length = cumulative_length[segment]
    selected_length = edge_length[segment].clamp_min(torch.finfo(controls.dtype).eps)
    fraction = ((distance - start_length) / selected_length).clamp(0.0, 1.0)
    sampled = controls[segment] + fraction[:, None] * edge[segment]

    sampled_edge = sampled[1:] - sampled[:-1]
    raw_heading = torch.atan2(sampled_edge[:, 1], sampled_edge[:, 0])
    wrapped_turn = torch.atan2(
        torch.sin(raw_heading[1:] - raw_heading[:-1]),
        torch.cos(raw_heading[1:] - raw_heading[:-1]),
    )
    unwrapped_heading = torch.cat((raw_heading[:1], raw_heading[:1] + wrapped_turn.cumsum(dim=0)))
    maximum_turn = 2.0 * math.asin(rest_length / (2.0 * _SUCCESS_ROUTE_MINIMUM_BEND_RADIUS))
    limited_heading = [unwrapped_heading[0]]
    for index in range(1, len(unwrapped_heading)):
        turn = (unwrapped_heading[index] - limited_heading[-1]).clamp(-maximum_turn, maximum_turn)
        limited_heading.append(limited_heading[-1] + turn)
    limited_heading = torch.stack(limited_heading)
    direction = torch.stack((torch.cos(limited_heading), torch.sin(limited_heading)), dim=-1)
    return torch.cat(
        (
            sampled[:1],
            sampled[:1] + torch.cumsum(rest_length * direction, dim=0),
        ),
        dim=0,
    )


def _build_success_cable_poses(
    peg_positions_w: torch.Tensor,
    board_center_xy_w: torch.Tensor,
    cable_z_w: torch.Tensor,
    num_segments: int,
    rest_length: float,
) -> torch.Tensor:
    """Construct a connected cable pose satisfying the medium task route."""
    import torch

    device = peg_positions_w.device
    dtype = peg_positions_w.dtype
    controls = [board_center_xy_w + torch.tensor(_SUCCESS_ROUTE_START, device=device, dtype=dtype)]
    exit_radial = torch.tensor((1.0, 0.0), device=device, dtype=dtype)

    for step, (peg_index, direction) in enumerate(_SUCCESS_ROUTE):
        center = peg_positions_w[peg_index, :2]
        if step == 0:
            entry_angle = torch.tensor(_SUCCESS_ROUTE_ENTRY_ANGLE, device=device, dtype=dtype)
        else:
            incoming = controls[-1] - center
            entry_angle = torch.atan2(incoming[1], incoming[0])
        angle = entry_angle - float(direction) * _SUCCESS_ROUTE_WINDING * torch.linspace(
            0.0,
            1.0,
            _SUCCESS_ROUTE_ARC_SAMPLES,
            device=device,
            dtype=dtype,
        )
        radial = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        arc = center[None] + _SUCCESS_ROUTE_WRAP_RADIUS * radial
        controls.extend((arc[0] + 0.040 * radial[0], *arc, arc[-1] + 0.040 * radial[-1]))
        exit_radial = radial[-1]

    board_half_size = torch.tensor(_SUCCESS_ROUTE_BOARD_HALF_SIZE, device=device, dtype=dtype)
    corner_offset = board_half_size - _SUCCESS_ROUTE_BOARD_MARGIN
    corners = board_center_xy_w + torch.stack((
        torch.stack((-corner_offset[0], -corner_offset[1])),
        torch.stack((corner_offset[0], -corner_offset[1])),
        torch.stack((corner_offset[0], corner_offset[1])),
        torch.stack((-corner_offset[0], corner_offset[1])),
    ))
    exit_point = controls[-1]
    corner_index = (1 if float(exit_radial[0]) >= 0.0 else 0) + (2 if float(exit_radial[1]) >= 0.0 else 0)
    corner_index = (0, 1, 3, 2)[corner_index]
    next_counterclockwise = (corner_index + 1) % 4
    next_clockwise = (corner_index - 1) % 4
    counterclockwise_direction = torch.nn.functional.normalize(
        corners[next_counterclockwise] - corners[corner_index], dim=0
    )
    clockwise_direction = torch.nn.functional.normalize(corners[next_clockwise] - corners[corner_index], dim=0)
    incoming_direction = torch.nn.functional.normalize(corners[corner_index] - exit_point, dim=0)
    perimeter_step = (
        1
        if float(torch.dot(counterclockwise_direction, incoming_direction))
        >= float(torch.dot(clockwise_direction, incoming_direction))
        else -1
    )
    perimeter = [corners[(corner_index + perimeter_step * index) % 4] for index in range(4)]
    controls.extend((exit_point + 0.030 * exit_radial, *perimeter))

    vertices = _sample_exact_length_curve(torch.stack(controls), num_segments, rest_length)
    direction = torch.nn.functional.normalize(vertices[1:] - vertices[:-1], dim=-1)
    first_quaternion = torch.nn.functional.normalize(
        torch.stack((-direction[0, 1], direction[0, 0], direction.new_zeros(()), direction.new_ones(()))),
        dim=0,
    )
    turn = torch.atan2(
        direction[:-1, 0] * direction[1:, 1] - direction[:-1, 1] * direction[1:, 0],
        (direction[:-1] * direction[1:]).sum(dim=-1),
    )
    cumulative_turn = torch.cat((turn.new_zeros(1), turn.cumsum(dim=0)))
    half_turn = 0.5 * cumulative_turn
    transport = torch.stack(
        (
            torch.zeros_like(half_turn),
            torch.zeros_like(half_turn),
            torch.sin(half_turn),
            torch.cos(half_turn),
        ),
        dim=-1,
    )
    quaternion = torch.nn.functional.normalize(
        _quat_mul_xyzw(transport, first_quaternion.expand_as(transport)),
        dim=-1,
    )
    position = torch.cat((vertices[:-1], cable_z_w.expand(num_segments, 1)), dim=-1)
    return torch.cat((position, quaternion), dim=-1)


@dataclass(frozen=True)
class MotionPhase:
    """One convergence-gated pair of absolute YAM joint targets."""

    label: str
    left_target: tuple[float, ...]
    right_target: tuple[float, ...]
    gripper_command: float
    action_limit: float
    minimum_steps: int
    maximum_steps: int
    tolerance: float | None


_LEFT_APPROACH = _mirror_yam_joints(_RIGHT_APPROACH)

_JOINT_MOTION_PHASES = (
    MotionPhase("settle", _HOME, _HOME, _GRIPPER_OPEN, 0.60, 15, 30, 0.06),
    MotionPhase(
        "approach opposite cable sections",
        _LEFT_APPROACH,
        _RIGHT_APPROACH,
        _GRIPPER_OPEN,
        0.30,
        70,
        180,
        0.03,
    ),
)

_RETURN_HOME_PHASE = MotionPhase(
    "return home",
    _HOME,
    _HOME,
    _GRIPPER_OPEN,
    0.30,
    70,
    180,
    0.04,
)


def _build_cable_demo_environment():
    """Compose the medium cable-routing environment used by this demo."""
    from isaaclab_arena_environments.isaac_cap.cable_routing.environment import (
        CableRoutingMediumEnvironment,
        CableRoutingMediumEnvironmentCfg,
    )

    return CableRoutingMediumEnvironment().build(CableRoutingMediumEnvironmentCfg())


class CableEnvBehaviourDemo(EnvBehaviourDemo):
    """Implement the cable-specific setup and bimanual motion sequence."""

    label = "cable-behaviour-demo"

    def __init__(
        self,
        simulation_app,
        arena_environment,
        builder_cfg,
        *,
        pause_steps: int,
        real_time: bool = True,
        visualizer_cfg=None,
    ) -> None:
        """Configure the cable behavior and success-state display duration."""
        assert pause_steps >= 1, "pause_steps must be positive."
        super().__init__(
            simulation_app,
            arena_environment,
            builder_cfg,
            real_time=real_time,
            visualizer_cfg=visualizer_cfg,
        )
        self.pause_steps = pause_steps

    def setup_demo(self) -> None:
        """Resolve the bimanual action interface and cable success geometry."""
        import torch

        assert self.env.action_space.shape == (1, 14), f"Unexpected action shape {self.env.action_space.shape}."
        self.torch = torch

        left_finger_ids, _ = self.base_env.scene["left_robot"].find_bodies(["lf_down", "rf_down"], preserve_order=True)
        right_finger_ids, _ = self.base_env.scene["right_robot"].find_bodies(
            ["lf_down", "rf_down"], preserve_order=True
        )
        assert len(left_finger_ids) == 2 and len(right_finger_ids) == 2, "Each YAM must have two lower fingers."
        self._finger_body_ids = (
            (left_finger_ids[0], left_finger_ids[1]),
            (right_finger_ids[0], right_finger_ids[1]),
        )
        left_wrist_ids, _ = self.base_env.scene["left_robot"].find_bodies(["link_6"])
        right_wrist_ids, _ = self.base_env.scene["right_robot"].find_bodies(["link_6"])
        assert len(left_wrist_ids) == len(right_wrist_ids) == 1, "Each YAM must have one link_6 body."
        self._wrist_body_ids = (left_wrist_ids[0], right_wrist_ids[0])
        self._wrist_quaternions = None

        success_cfg = self.base_env.termination_manager.get_term_cfg("success")
        self.success_term = success_cfg.func
        self.success_params = success_cfg.params
        self.cable = self.base_env.scene[self.success_params["cable_asset_name"]]
        self.peg_names = tuple(self.success_params["peg_asset_names"])
        configured_route = tuple(
            zip(
                self.success_params["route_peg_indices"],
                self.success_params["route_directions"],
                strict=True,
            )
        )
        assert (
            configured_route == _SUCCESS_ROUTE
        ), f"The success-pose generator supports {_SUCCESS_ROUTE}, got {configured_route}."

        default_positions = self.cable.data.default_segment_pose_w.torch[0, :, :3]
        segment_lengths = torch.linalg.vector_norm(default_positions[1:] - default_positions[:-1], dim=-1)
        self.cable_rest_length = float(segment_lengths.median())

    def _phase_action(self, phase: MotionPhase):
        """Return smooth absolute joint targets and the largest live target error."""
        left_joint_pos = self.base_env.scene["left_robot"].data.joint_pos.torch[:, :6]
        right_joint_pos = self.base_env.scene["right_robot"].data.joint_pos.torch[:, :6]
        left_target = self.torch.as_tensor(
            phase.left_target,
            device=self.base_env.device,
            dtype=left_joint_pos.dtype,
        ).expand_as(left_joint_pos)
        right_target = self.torch.as_tensor(
            phase.right_target,
            device=self.base_env.device,
            dtype=right_joint_pos.dtype,
        ).expand_as(right_joint_pos)
        maximum_delta = _MAX_ARM_JOINT_STEP * phase.action_limit

        action = self.torch.zeros(self.env.action_space.shape, device=self.base_env.device)
        action[:, :6] = left_joint_pos + (left_target - left_joint_pos).clamp(-maximum_delta, maximum_delta)
        action[:, 6] = phase.gripper_command
        action[:, 7:13] = right_joint_pos + (right_target - right_joint_pos).clamp(-maximum_delta, maximum_delta)
        action[:, 13] = phase.gripper_command
        maximum_error = self.torch.maximum(
            self.torch.abs(left_target - left_joint_pos).max(),
            self.torch.abs(right_target - right_joint_pos).max(),
        )
        return action, float(maximum_error)

    def _run_phase(self, phase: MotionPhase) -> None:
        """Run one convergence-gated motion phase."""
        print(f"[cable-behaviour-demo] {phase.label}", flush=True)
        for step in range(phase.maximum_steps):
            action, maximum_error = self._phase_action(phase)
            self.step(action)
            minimum_reached = step + 1 >= phase.minimum_steps
            target_reached = phase.tolerance is None or maximum_error <= phase.tolerance
            if minimum_reached and target_reached:
                return

    def _body_position_jacobian(self, asset, body_ids: tuple[int, ...]):
        """Return the arm-joint Jacobian for the selected body-position average."""
        assert all(body_id > 0 for body_id in body_ids), "The fixed articulation root has no Jacobian entry."
        jacobians = self.torch.stack(
            [asset.data.body_link_jacobian_w.torch[:, body_id - 1, :, :6] for body_id in body_ids],
            dim=0,
        )
        return jacobians.mean(dim=0)

    def _cartesian_arm_action(
        self,
        asset,
        body_ids: tuple[int, ...],
        target,
        joint_seed,
        action_limit: float,
        desired_quaternion,
        orientation_body_id: int,
    ):
        """Return damped-IK targets that move the fingers while holding the wrist orientation."""
        from isaaclab.utils.math import compute_pose_error

        position = asset.data.body_link_pos_w.torch[:, body_ids].mean(dim=1)
        position_error = target - position
        position_jacobian = self._body_position_jacobian(asset, body_ids)[:, :3]

        current_quaternion = asset.data.body_link_quat_w.torch[:, orientation_body_id]
        target_quaternion = desired_quaternion.expand_as(current_quaternion)
        target_quaternion = self.torch.where(
            ((current_quaternion * target_quaternion).sum(dim=-1) < 0.0).unsqueeze(-1),
            -target_quaternion,
            target_quaternion,
        )
        _, orientation_error = compute_pose_error(
            position,
            current_quaternion,
            position,
            target_quaternion,
        )
        orientation_weight = 0.30
        orientation_jacobian = asset.data.body_link_jacobian_w.torch[:, orientation_body_id - 1, 3:6, :6]
        task_error = self.torch.cat((position_error, orientation_weight * orientation_error), dim=-1)
        jacobian = self.torch.cat((position_jacobian, orientation_weight * orientation_jacobian), dim=1)
        jacobian_t = jacobian.transpose(-1, -2)
        task_identity = self.torch.eye(jacobian.shape[1], device=position.device, dtype=position.dtype).unsqueeze(0)
        pseudo_inverse = jacobian_t @ self.torch.linalg.solve(
            jacobian @ jacobian_t + 0.025**2 * task_identity,
            task_identity,
        )

        joint_positions = asset.data.joint_pos.torch[:, :6]
        task_delta = (pseudo_inverse @ task_error.unsqueeze(-1)).squeeze(-1)
        joint_identity = self.torch.eye(6, device=position.device, dtype=position.dtype).unsqueeze(0)
        null_space = joint_identity - pseudo_inverse @ jacobian
        posture_delta = (null_space @ (joint_seed - joint_positions).unsqueeze(-1)).squeeze(-1)
        maximum_delta = _MAX_ARM_JOINT_STEP * action_limit
        action = joint_positions + (0.65 * task_delta + 0.08 * posture_delta).clamp(
            -maximum_delta,
            maximum_delta,
        )
        error = self.torch.linalg.vector_norm(position_error, dim=-1).max()
        return action, float(error)

    def _run_cartesian_phase(
        self,
        label: str,
        left_target,
        right_target,
        left_gripper_command: float,
        right_gripper_command: float,
        *,
        action_limit: float,
        minimum_steps: int,
        maximum_steps: int,
        tolerance: float | None,
        left_gripper_start_command: float | None = None,
        right_gripper_start_command: float | None = None,
    ) -> None:
        """Move both lower-finger midpoints toward world-space targets."""
        assert self._wrist_quaternions is not None, "Capture the tool-down wrist orientations before Cartesian motion."
        print(f"[cable-behaviour-demo] {label}", flush=True)
        left = self.base_env.scene["left_robot"]
        right = self.base_env.scene["right_robot"]
        left_seed = left.data.joint_pos.torch[:, :6].clone()
        right_seed = right.data.joint_pos.torch[:, :6].clone()

        for step in range(maximum_steps):
            action = self.torch.zeros(self.env.action_space.shape, device=self.base_env.device)
            left_action, left_error = self._cartesian_arm_action(
                left,
                self._finger_body_ids[0],
                left_target,
                left_seed,
                action_limit,
                self._wrist_quaternions[0],
                self._wrist_body_ids[0],
            )
            right_action, right_error = self._cartesian_arm_action(
                right,
                self._finger_body_ids[1],
                right_target,
                right_seed,
                action_limit,
                self._wrist_quaternions[1],
                self._wrist_body_ids[1],
            )
            progress = min((step + 1) / minimum_steps, 1.0)
            progress = progress * progress * (3.0 - 2.0 * progress)
            live_left_gripper_command = left_gripper_command
            if left_gripper_start_command is not None:
                live_left_gripper_command = left_gripper_start_command + progress * (
                    left_gripper_command - left_gripper_start_command
                )
            live_right_gripper_command = right_gripper_command
            if right_gripper_start_command is not None:
                live_right_gripper_command = right_gripper_start_command + progress * (
                    right_gripper_command - right_gripper_start_command
                )
            action[:, :6] = left_action
            action[:, 6] = live_left_gripper_command
            action[:, 7:13] = right_action
            action[:, 13] = live_right_gripper_command
            self.step(action)

            minimum_reached = step + 1 >= minimum_steps
            target_reached = tolerance is None or max(left_error, right_error) <= tolerance
            if minimum_reached and target_reached:
                return

    def _finger_midpoints(self):
        """Return the live lower-finger midpoint of each YAM."""
        left = self.base_env.scene["left_robot"].data.body_link_pos_w.torch[:, self._finger_body_ids[0]].mean(dim=1)
        right = self.base_env.scene["right_robot"].data.body_link_pos_w.torch[:, self._finger_body_ids[1]].mean(dim=1)
        return left.clone(), right.clone()

    def _cable_targets(self, clearance: float):
        """Return gripper targets above the live cable sections selected for pinching."""
        cable_positions = self.cable.data.segment_pose_w.torch[:, :, :3]
        left_target = cable_positions[:, _LEFT_CABLE_SEGMENT].clone()
        right_target = cable_positions[:, _RIGHT_CABLE_SEGMENT].clone()
        left_target[:, 2] += clearance
        right_target[:, 2] += clearance
        return left_target, right_target

    def _right_drag_waypoints(self, right_start):
        """Build the shortened, peg-clear portion of the original right-gripper drag."""
        peg_xy = self.torch.stack(
            [self.base_env.scene[name].data.root_pos_w.torch[:, :2] for name in self.peg_names],
            dim=1,
        )
        route_peg_xy = peg_xy[:, 0]
        radial = right_start[:, :2] - route_peg_xy
        radial = radial / self.torch.linalg.vector_norm(radial, dim=-1, keepdim=True).clamp_min(1.0e-6)
        entry = right_start.clone()
        entry[:, :2] = route_peg_xy + _DRAG_RADIUS * radial

        waypoints = [
            right_start + (entry - right_start) * (step / _DRAG_APPROACH_STEPS)
            for step in range(1, _DRAG_APPROACH_STEPS + 1)
        ]
        path = self.torch.stack((right_start, *waypoints), dim=1)
        segment_start = path[:, :-1, :2]
        segment_delta = path[:, 1:, :2] - segment_start
        peg_from_start = peg_xy[:, None] - segment_start[:, :, None]
        denominator = (segment_delta * segment_delta).sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        fraction = (peg_from_start * segment_delta[:, :, None]).sum(dim=-1) / denominator
        fraction = fraction.clamp(0.0, 1.0)
        nearest_point = segment_start[:, :, None] + fraction[..., None] * segment_delta[:, :, None]
        peg_clearance = self.torch.linalg.vector_norm(peg_xy[:, None] - nearest_point, dim=-1)
        assert bool(
            (peg_clearance >= _GRIPPER_PEG_CLEARANCE).all().item()
        ), f"Right-gripper path comes within {float(peg_clearance.min()):.3f} m of a peg."
        return waypoints

    def _hold_action(self):
        """Hold both YAMs at their current arm positions with open grippers."""
        action = self.torch.zeros(self.env.action_space.shape, device=self.base_env.device)
        action[:, :6] = self.base_env.scene["left_robot"].data.joint_pos.torch[:, :6]
        action[:, 7:13] = self.base_env.scene["right_robot"].data.joint_pos.torch[:, :6]
        action[:, 6] = _GRIPPER_OPEN
        action[:, 13] = _GRIPPER_OPEN
        return action

    def _successful_cable_pose(self):
        """Return a physically connected cable pose satisfying the configured route."""
        peg_positions_w = self.torch.stack(
            [self.base_env.scene[name].data.root_pos_w.torch[0] for name in self.peg_names]
        )
        board_center_xy_w = self.base_env.scene["board"].data.root_pos_w.torch[0, :2]
        cable_z_w = self.cable.data.default_segment_pose_w.torch[0, :, 2].mean()
        pose = _build_success_cable_poses(
            peg_positions_w,
            board_center_xy_w,
            cable_z_w,
            self.cable.num_segments,
            self.cable_rest_length,
        ).unsqueeze(0)

        from isaaclab_arena_environments.isaac_cap.cable_routing.geometry import cable_route_success_from_geometry

        success = cable_route_success_from_geometry(
            pose[:, :, :3],
            peg_positions_w.unsqueeze(0),
            route_peg_indices=self.success_params["route_peg_indices"],
            route_directions=self.success_params["route_directions"],
        )
        assert bool(success.all().item()), "Generated cable pose does not satisfy the task's success predicate."
        return pose

    def _teleport_cable_to_success(self) -> None:
        """Write a valid route through the native cable state API."""
        pose = self._successful_cable_pose()
        env_ids = self.torch.arange(self.base_env.num_envs, device=self.base_env.device, dtype=self.torch.int32)
        self.cable.write_segment_pose_to_sim_index(segment_pose=pose, env_ids=env_ids)
        self.cable.write_segment_velocity_to_sim_index(
            segment_velocity=self.torch.zeros(
                (self.base_env.num_envs, self.cable.num_segments, 6),
                device=self.base_env.device,
            ),
            env_ids=env_ids,
        )
        success = self.success_term(self.base_env, **self.success_params)
        assert bool(success.all().item()), "Teleported cable state was not recognized as successful."

    def _step_hold(self):
        """Advance once while holding the arms and return the ended-environment mask."""
        _, _, terminated, truncated, _ = self.step(self._hold_action())
        return terminated | truncated

    def run_cycle(self, cycle: int) -> None:
        """Gently manipulate the cable, then demonstrate success and automatic reset."""
        initial_cable_position = self.base_env.scene["cable"].data.segment_pose_w.torch[..., :3].clone()
        print(f"[cable-behaviour-demo] starting cycle {cycle}", flush=True)
        for phase in _JOINT_MOTION_PHASES:
            self._run_phase(phase)

        left = self.base_env.scene["left_robot"]
        right = self.base_env.scene["right_robot"]
        # The known approach poses are tool-down; preserve both wrist orientations during IK.
        self._wrist_quaternions = (
            left.data.body_link_quat_w.torch[:, self._wrist_body_ids[0]].clone(),
            right.data.body_link_quat_w.torch[:, self._wrist_body_ids[1]].clone(),
        )

        high_hover_targets = self._cable_targets(_HIGH_HOVER_CLEARANCE)
        self._run_cartesian_phase(
            "move both grippers to a high hover above the cable",
            *high_hover_targets,
            _GRIPPER_OPEN,
            _GRIPPER_OPEN,
            action_limit=0.30,
            minimum_steps=50,
            maximum_steps=180,
            tolerance=0.008,
        )

        low_hover_targets = self._cable_targets(_LOW_HOVER_CLEARANCE)
        self._run_cartesian_phase(
            "descend both grippers to a low hover",
            *low_hover_targets,
            _GRIPPER_OPEN,
            _GRIPPER_OPEN,
            action_limit=0.20,
            minimum_steps=45,
            maximum_steps=140,
            tolerance=0.005,
        )

        grasp_targets = self._cable_targets(_GRASP_HEIGHT_OFFSET)
        self._run_cartesian_phase(
            "lower both grippers gently around the cable",
            *grasp_targets,
            _GRIPPER_OPEN,
            _GRIPPER_OPEN,
            action_limit=0.12,
            minimum_steps=50,
            maximum_steps=140,
            tolerance=0.004,
        )
        self._run_cartesian_phase(
            "close both grippers around the cable",
            *grasp_targets,
            _GRIPPER_CLOSED,
            _GRIPPER_CLOSED,
            action_limit=0.10,
            minimum_steps=75,
            maximum_steps=75,
            tolerance=None,
            left_gripper_start_command=_GRIPPER_OPEN,
            right_gripper_start_command=_GRIPPER_OPEN,
        )

        lift_targets = self._cable_targets(_DRAG_HEIGHT_OFFSET)
        self._run_cartesian_phase(
            "lift the grasped cable slightly",
            *lift_targets,
            _GRIPPER_CLOSED,
            _GRIPPER_CLOSED,
            action_limit=0.15,
            minimum_steps=55,
            maximum_steps=140,
            tolerance=0.005,
        )

        left_hold, right_start = self._finger_midpoints()
        right_drag_waypoints = self._right_drag_waypoints(right_start)
        for waypoint_index, right_waypoint in enumerate(right_drag_waypoints, start=1):
            self._run_cartesian_phase(
                f"short peg-clear right cable drag ({waypoint_index}/{len(right_drag_waypoints)})",
                left_hold,
                right_waypoint,
                _GRIPPER_CLOSED,
                _GRIPPER_CLOSED,
                action_limit=0.14,
                minimum_steps=14,
                maximum_steps=70,
                tolerance=0.006,
            )

        release_targets = self._finger_midpoints()
        self._run_cartesian_phase(
            "release the cable",
            *release_targets,
            _GRIPPER_OPEN,
            _GRIPPER_OPEN,
            action_limit=0.10,
            minimum_steps=45,
            maximum_steps=45,
            tolerance=None,
            left_gripper_start_command=_GRIPPER_CLOSED,
            right_gripper_start_command=_GRIPPER_CLOSED,
        )

        release_targets = self._finger_midpoints()
        retreat_targets = tuple(
            target + target.new_tensor((0.0, 0.0, _RETREAT_CLEARANCE)) for target in release_targets
        )
        self._run_cartesian_phase(
            "retreat vertically from the cable",
            *retreat_targets,
            _GRIPPER_OPEN,
            _GRIPPER_OPEN,
            action_limit=0.25,
            minimum_steps=50,
            maximum_steps=140,
            tolerance=0.008,
        )
        self._run_phase(_RETURN_HOME_PHASE)

        final_cable_position = self.base_env.scene["cable"].data.segment_pose_w.torch[..., :3]
        displacement = self.torch.linalg.vector_norm(final_cable_position - initial_cable_position, dim=-1).max()
        print(
            f"[cable-behaviour-demo] cycle {cycle} physically moved the cable by up to {float(displacement):.3f} m",
            flush=True,
        )

        print(f"[cable-behaviour-demo] cycle {cycle}: teleport cable to the successful route", flush=True)
        self._teleport_cable_to_success()
        for _ in range(self.pause_steps):
            self.render()

        reset_observed = self._step_hold()
        if not bool(reset_observed.all().item()):
            raise RuntimeError("Successful cable route did not trigger the environment reset.")
        print(f"[cable-behaviour-demo] cycle {cycle}: success reset observed", flush=True)

        for _ in range(self.pause_steps):
            if bool(self._step_hold().any().item()):
                raise RuntimeError("Environment ended unexpectedly while displaying the reset state.")


def run_demo(
    simulation_app,
    *,
    cycles: int = 0,
    pause_steps: int = 30,
    real_time: bool = True,
) -> None:
    """Compose cable routing and run its behavior through the shared lifecycle."""
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = CableEnvBehaviourDemo(
        simulation_app,
        _build_cable_demo_environment(),
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False),
        pause_steps=pause_steps,
        real_time=real_time,
    )
    demo.run_demo(cycles)


def main() -> None:
    """Launch the visual cable behavior demo."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=0, help="Cycles to run; zero repeats until Kit closes.")
    parser.add_argument(
        "--pause-steps",
        type=int,
        default=30,
        help="Frames shown for the teleported success and reset states.",
    )
    parser.add_argument("--no-real-time", action="store_true", help="Run without wall-clock rate limiting.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    args.limit_cpu_threads = 1

    with SimulationAppContext(args) as simulation_app:
        run_demo(
            simulation_app,
            cycles=args.cycles,
            pause_steps=args.pause_steps,
            real_time=not args.no_real_time,
        )


if __name__ == "__main__":
    main()
