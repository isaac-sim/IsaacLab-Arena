# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioral gripper interfaces and robot-specific implementations."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


_ROBOTIQ_2F85_LINKAGE_AMPLITUDE_M = 0.1143
_ROBOTIQ_2F85_PHASE_OFFSET_RAD = 0.715
_ROBOTIQ_2F85_GAP_OFFSET_M = 0.01


class Gripper(Protocol):
    """Gripper state exposed to embodiment-agnostic tasks."""

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        """Return the gripper position in world frame with shape ``(num_envs, 3)``."""
        ...

    def get_opening_width_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the gripper opening width in meters with shape ``(num_envs,)``."""
        ...


class ParallelJawGripper(Gripper, Protocol):
    """Gripper whose grasp state is represented by a physical jaw gap."""

    def get_jaw_gap_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the physical jaw gap in meters with shape ``(num_envs,)``."""
        ...

    def get_opening_width_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the jaw gap as the gripper opening width."""
        return self.get_jaw_gap_m(world)


@dataclass(frozen=True, kw_only=True)
class PandaGripper(ParallelJawGripper):
    """Franka Panda parallel-jaw gripper."""

    left_finger_joint_name: str = "panda_finger_joint1"
    """Joint measuring the left finger's distance from the centerline."""

    right_finger_joint_name: str = "panda_finger_joint2"
    """Joint measuring the right finger's distance from the centerline."""

    frame_transformer_name: str = "ee_frame"
    """Scene key of the gripper frame transformer."""

    target_frame_name: str = "end_effector"
    """Frame-transformer target representing the gripper center."""

    def get_jaw_gap_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the sum of the two prismatic finger positions."""
        left = world.get_joint_position("robot", self.left_finger_joint_name)
        right = world.get_joint_position("robot", self.right_finger_joint_name)
        return left + right

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        """Return the configured Franka grasp-frame position."""
        return world.get_frame_position_w(self.frame_transformer_name, self.target_frame_name)


@dataclass(frozen=True, kw_only=True)
class RobotiqGripper(ParallelJawGripper):
    """Robotiq 2F-85 gripper with independently configurable measurements.

    Joint and frame choices select simulation data sources for the same physical
    gripper; they do not represent distinct gripper types.
    """

    driver_joint_name: str | None = None
    """Joint used to measure opening, or None to use tracked finger pads."""

    body_name: str | None = None
    """Body used to measure position, or None to use the frame-transformer target."""

    body_point_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Gripper point offset in the configured body frame; requires body_name."""

    frame_transformer_name: str = "ee_frame"
    """Scene key of the Robotiq frame transformer."""

    target_frame_name: str = "end_effector"
    """Frame-transformer target representing the gripper center."""

    left_finger_frame_name: str = "tool_leftfinger"
    """Frame-transformer target on the left finger pad."""

    right_finger_frame_name: str = "tool_rightfinger"
    """Frame-transformer target on the right finger pad."""

    def __post_init__(self) -> None:
        """Validate the body-relative position measurement settings."""
        assert len(self.body_point_offset_xyz) == 3, "body_point_offset_xyz must contain three values."
        assert all(
            math.isfinite(value) for value in self.body_point_offset_xyz
        ), "body_point_offset_xyz must contain only finite values."
        assert self.body_name is not None or all(
            value == 0.0 for value in self.body_point_offset_xyz
        ), "body_point_offset_xyz requires body_name when nonzero."

    def get_jaw_gap_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the physical distance between the two finger pads."""
        if self.driver_joint_name is not None:
            driver_position = world.get_joint_position("robot", self.driver_joint_name)
            # The 2F-85 linkage maps its driver angle to an 85 mm maximum inner-finger opening.
            jaw_gap_m = (
                _ROBOTIQ_2F85_LINKAGE_AMPLITUDE_M * torch.sin(_ROBOTIQ_2F85_PHASE_OFFSET_RAD - driver_position)
                + _ROBOTIQ_2F85_GAP_OFFSET_M
            )
            return torch.clamp(jaw_gap_m, min=0.0, max=0.085)
        left = world.get_frame_position_w(self.frame_transformer_name, self.left_finger_frame_name)
        right = world.get_frame_position_w(self.frame_transformer_name, self.right_finger_frame_name)
        return torch.linalg.vector_norm(left - right, dim=-1)

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        """Return the configured Robotiq grasp-frame position."""
        if self.body_name is not None:
            T_W_B = world.get_body_pose_w("robot", self.body_name)
            t_B_G = T_W_B.new_tensor(self.body_point_offset_xyz).expand_as(T_W_B[:, :3])
            t_W_G, _ = combine_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], t_B_G)
            return t_W_G
        return world.get_frame_position_w(self.frame_transformer_name, self.target_frame_name)


__all__ = ["Gripper", "PandaGripper", "ParallelJawGripper", "RobotiqGripper"]
