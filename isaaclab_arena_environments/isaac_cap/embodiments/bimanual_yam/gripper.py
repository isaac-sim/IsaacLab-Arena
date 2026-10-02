# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bimanual YAM gripper implementation."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.embodiments.gripper import ParallelJawGripper

from .config import GRIPPER_JOINT_NAME

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


@dataclass(kw_only=True)
class YamGripper(ParallelJawGripper):
    """YAM parallel-jaw gripper used by the right work arm."""

    articulation_name: str = "right_robot"
    """Scene key of the work-arm articulation."""

    driver_joint_name: str = GRIPPER_JOINT_NAME
    """Joint measuring one finger's distance from the centerline."""

    action_term_name: str = "right_gripper_action"
    """Action term supplying the driven finger's commanded position."""

    frame_transformer_name: str = "right_ee_frame"
    """Scene key of the work-arm frame transformer."""

    target_frame_name: str = "tcp"
    """Frame-transformer target representing the work-hand TCP."""

    def get_jaw_gap_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return twice the driven finger displacement as the physical jaw gap."""
        return 2.0 * world.get_joint_position(self.articulation_name, self.driver_joint_name)

    def get_closing_error_m(self, env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
        """Return the driven finger's measured-minus-commanded displacement."""
        measured = env.arena_world.get_joint_position(self.articulation_name, self.driver_joint_name)
        commanded = env.action_manager.get_term(self.action_term_name).processed_actions[:, 0]
        return measured - commanded

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        """Return the tracked work-hand TCP position."""
        return world.get_frame_position_w(self.frame_transformer_name, self.target_frame_name)


__all__ = ["YamGripper"]
