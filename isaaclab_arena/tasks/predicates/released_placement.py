# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Released, stable destination-contact and container-placement predicates."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import warp as wp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def released_contact_condition(
    object_position_w: torch.Tensor,
    object_velocity_w: torch.Tensor,
    destination_position_w: torch.Tensor,
    gripper_joint_position: torch.Tensor,
    end_effector_position_w: torch.Tensor,
    destination_contact_force: torch.Tensor,
    *,
    max_horizontal_offset: float,
    max_linear_speed: float,
    max_angular_speed: float,
    max_open_joint_position: float,
    min_end_effector_distance: float,
    min_contact_force: float,
) -> torch.Tensor:
    """Return a released, withdrawn, stable version of Arena's destination-contact condition.

    This deliberately does not prescribe object height or orientation. Those are properties of a
    stricter container-placement label, while Arena's published pick-and-place objective is stable
    contact with the destination.
    """

    horizontal_offset = torch.linalg.vector_norm(
        (object_position_w - destination_position_w)[:, :2], dim=-1
    )
    linear_speed = torch.linalg.vector_norm(object_velocity_w[:, :3], dim=-1)
    angular_speed = torch.linalg.vector_norm(object_velocity_w[:, 3:], dim=-1)
    end_effector_distance = torch.linalg.vector_norm(end_effector_position_w - object_position_w, dim=-1)

    return (
        (horizontal_offset <= max_horizontal_offset)
        & (linear_speed <= max_linear_speed)
        & (angular_speed <= max_angular_speed)
        & (gripper_joint_position <= max_open_joint_position)
        & (end_effector_distance >= min_end_effector_distance)
        & (destination_contact_force > min_contact_force)
    )


def released_placement_condition(
    object_pose_w: torch.Tensor,
    object_velocity_w: torch.Tensor,
    destination_position_w: torch.Tensor,
    gripper_joint_position: torch.Tensor,
    end_effector_position_w: torch.Tensor,
    destination_contact_force: torch.Tensor,
    *,
    max_horizontal_offset: float,
    min_vertical_offset: float,
    max_vertical_offset: float,
    max_axis_tilt: float,
    max_linear_speed: float,
    max_angular_speed: float,
    max_open_joint_position: float,
    min_end_effector_distance: float,
    min_contact_force: float,
) -> torch.Tensor:
    """Return the per-environment released-placement condition.

    Args:
        object_pose_w: Object root pose ``[x, y, z, qx, qy, qz, qw]`` in world coordinates.
        object_velocity_w: Object root velocity ``[vx, vy, vz, wx, wy, wz]``.
        destination_position_w: Destination root position in world coordinates [m].
        gripper_joint_position: Physical gripper joint position [rad].
        end_effector_position_w: Gripper-base position in world coordinates [m].
        destination_contact_force: Object-to-destination contact-force magnitude [N].
        max_horizontal_offset: Maximum object/destination center offset in XY [m].
        min_vertical_offset: Minimum object-center height relative to destination [m].
        max_vertical_offset: Maximum object-center height relative to destination [m].
        max_axis_tilt: Maximum angle between the object's local Z axis and world vertical [rad].
        max_linear_speed: Maximum object linear speed [m/s].
        max_angular_speed: Maximum object angular speed [rad/s].
        max_open_joint_position: Largest gripper joint position considered open [rad].
        min_end_effector_distance: Minimum object-center to gripper-base distance [m].
        min_contact_force: Minimum object-to-destination contact force [N].

    Returns:
        Boolean tensor shaped ``(num_envs,)``.
    """

    object_position_w = object_pose_w[:, :3]
    object_quaternion_xyzw = object_pose_w[:, 3:]
    object_quaternion_xyzw = object_quaternion_xyzw / torch.linalg.vector_norm(
        object_quaternion_xyzw, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(object_pose_w.dtype).eps)

    relative_position = object_position_w - destination_position_w
    horizontal_offset = torch.linalg.vector_norm(relative_position[:, :2], dim=-1)
    vertical_offset = relative_position[:, 2]

    # The local Z axis has world-Z component 1 - 2(qx^2 + qy^2). The absolute value treats
    # upright and upside-down placements equally because the task does not prescribe box polarity.
    qx, qy = object_quaternion_xyzw[:, 0], object_quaternion_xyzw[:, 1]
    vertical_alignment = torch.abs(1.0 - 2.0 * (qx.square() + qy.square()))
    sufficiently_upright = vertical_alignment >= torch.cos(
        torch.as_tensor(max_axis_tilt, dtype=object_pose_w.dtype, device=object_pose_w.device)
    )

    linear_speed = torch.linalg.vector_norm(object_velocity_w[:, :3], dim=-1)
    angular_speed = torch.linalg.vector_norm(object_velocity_w[:, 3:], dim=-1)
    end_effector_distance = torch.linalg.vector_norm(end_effector_position_w - object_position_w, dim=-1)

    return (
        (horizontal_offset <= max_horizontal_offset)
        & (vertical_offset >= min_vertical_offset)
        & (vertical_offset <= max_vertical_offset)
        & sufficiently_upright
        & (linear_speed <= max_linear_speed)
        & (angular_speed <= max_angular_speed)
        & (gripper_joint_position <= max_open_joint_position)
        & (end_effector_distance >= min_end_effector_distance)
        & (destination_contact_force >= min_contact_force)
    )


def update_consecutive_true_counts(counts: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """Increment counts where condition is true and clear them elsewhere."""

    return torch.where(condition, counts + 1, torch.zeros_like(counts))


class ReleasedPlacementDwell(ManagerTermBase):
    """Terminate after released placement remains valid for consecutive control steps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._consecutive_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._consecutive_steps[ids] = 0

    def __call__(
        self,
        env,
        object_cfg: SceneEntityCfg,
        destination_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg,
        max_horizontal_offset: float,
        min_vertical_offset: float,
        max_vertical_offset: float,
        max_axis_tilt: float,
        max_linear_speed: float,
        max_angular_speed: float,
        max_open_joint_position: float,
        min_end_effector_distance: float,
        min_contact_force: float,
        dwell_steps: int,
    ) -> torch.Tensor:
        """Return true after the configured released-placement dwell."""

        unwrapped_env = env.unwrapped
        object_entity = unwrapped_env.scene[object_cfg.name]
        destination_entity = unwrapped_env.scene[destination_cfg.name]
        contact_sensor = unwrapped_env.scene[contact_sensor_cfg.name]
        robot = unwrapped_env.scene[robot_cfg.name]

        assert len(robot_cfg.joint_ids) == 1, "ReleasedPlacementDwell requires exactly one gripper joint."
        assert len(robot_cfg.body_ids) == 1, "ReleasedPlacementDwell requires exactly one end-effector body."
        assert contact_sensor.data.force_matrix_w.shape[1:3] == (
            1,
            1,
        ), "ReleasedPlacementDwell requires one sensor body and one filtered destination body."

        object_pose_w = wp.to_torch(object_entity.data.root_link_pose_w)
        object_velocity_w = wp.to_torch(object_entity.data.root_link_vel_w)
        destination_position_w = wp.to_torch(destination_entity.data.root_link_pose_w)[:, :3]
        gripper_joint_position = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids].squeeze(-1)
        end_effector_position_w = wp.to_torch(robot.data.body_pos_w)[:, robot_cfg.body_ids, :].squeeze(1)
        destination_contact_force = torch.linalg.vector_norm(
            wp.to_torch(contact_sensor.data.force_matrix_w), dim=-1
        ).reshape(-1)

        condition = released_placement_condition(
            object_pose_w=object_pose_w,
            object_velocity_w=object_velocity_w,
            destination_position_w=destination_position_w,
            gripper_joint_position=gripper_joint_position,
            end_effector_position_w=end_effector_position_w,
            destination_contact_force=destination_contact_force,
            max_horizontal_offset=max_horizontal_offset,
            min_vertical_offset=min_vertical_offset,
            max_vertical_offset=max_vertical_offset,
            max_axis_tilt=max_axis_tilt,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            max_open_joint_position=max_open_joint_position,
            min_end_effector_distance=min_end_effector_distance,
            min_contact_force=min_contact_force,
        )
        self._consecutive_steps = update_consecutive_true_counts(self._consecutive_steps, condition)
        return (self._consecutive_steps >= dwell_steps).clone()
