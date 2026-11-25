# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 inspection, pickup and first-choice tracking at a four-lane loading station."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import (
    ARMS_CARRY,
    ARMS_PLACE,
    ARMS_SQUEEZE,
    ARMS_WIDE,
    G1Waypoint,
)
from isaaclab_arena_environments.spatial_memory.parts import board, box_item
from isaaclab_arena_environments.spatial_memory.tasks.box_carry import (
    LOAD_DEPTH_M,
    LOAD_WIDTH_M,
    TABLE_SIZE_XY_M,
    BoxCarryLogic,
)

LANE_PITCH_M = 0.92
"""Lateral distance between adjacent lane centers."""
COMMIT_X_M = -0.77
"""Environment-frame x coordinate of the first lane choice after inspection."""


class LoadingLaneLogic(BoxCarryLogic):
    """Require the first lane choice to match the inspected layout."""

    forbids_searching_again = True
    reobservation_budget_s = 0.0

    def memory_target_names(self):
        return self.obstacle_names()

    def pickup_parts(self):
        """Build the loading table and its graspable carton."""
        return [
            board("pick_table", (*TABLE_SIZE_XY_M, self.table_height())),
            box_item("load", (LOAD_DEPTH_M, LOAD_WIDTH_M, self.load_height), "orange", 0.25),
        ]

    def on_reset(self, env, episode, env_ids):
        """Place the G1 at the sampled inspection pose."""
        robot = env.scene["robot"]
        T_W_pelvis = robot.data.root_pose_w.torch[env_ids].clone()
        T_W_pelvis[:, :2] = env.scene.env_origins[env_ids, :2] + episode.layout_params["t_E_start_xy"][env_ids]
        # Start the survey at one end of the rack and sweep across all four lanes.
        yaw = math.pi - 0.75
        T_W_pelvis[:, 3:] = T_W_pelvis.new_tensor((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
        robot.write_root_pose_to_sim(T_W_pelvis, env_ids=env_ids)

    def instruction_dim(self):
        return 1

    def instruction(self, env, episode):
        return episode.instruction_revealed[:, None].float()

    def _pelvis_xy_e(self, env) -> torch.Tensor:
        """Return the horizontal pelvis position in each environment frame."""
        return self.embodiment.tracked_pose_w(env)[:, :2] - env.scene.env_origins[:, :2]

    def hidden_phase(self, env, episode):
        return episode.instruction_revealed & ~self.commitment_made(env, episode, env.episode_length_buf > 0)

    def hidden_target_mask(self, env, episode, in_hidden_phase):
        return in_hidden_phase[:, None].expand(-1, len(self.obstacle_names()))

    def commitment_made(self, env, episode, active):
        return episode.task_state_buffer("crossed_line", False, torch.bool)

    def inside_lane_entrance(self, t_E_pelvis_xy):
        """Return whether the pelvis has passed the loading-side entrance line."""
        return t_E_pelvis_xy[:, 0] < COMMIT_X_M

    def on_step(self, env, episode, visible, active):
        """Reveal the query at pickup and latch the first crossing of the lane boundary."""
        t_E_pelvis_xy = self._pelvis_xy_e(env)
        at_pickup = t_E_pelvis_xy.norm(dim=-1) < 0.12
        surveyed = episode.visible_before_hidden_phase.all(dim=1)
        self.update_memory_probe(
            env, episode, visible, active, reveal_allowed=at_pickup & surveyed & episode.gripper_closed
        )
        committed = self.commitment_made(env, episode, active)
        crossing = active & ~committed & self.inside_lane_entrance(t_E_pelvis_xy)
        correct = episode.task_state_buffer("correct_lane", False, torch.bool)
        correct[crossing] = (
            episode.instruction_revealed[crossing]
            & episode.visible_before_hidden_phase.all(dim=1)[crossing]
            & ((t_E_pelvis_xy[:, 1] - episode.layout_params["t_E_target_y"]).abs()[crossing] < 0.28)
        )
        committed |= crossing

    def extra_failures(self, env, episode):
        crossed = episode.task_state_buffer("crossed_line", False, torch.bool)
        correct = episode.task_state_buffer("correct_lane", False, torch.bool)
        return super().extra_failures(env, episode) | (crossed & ~correct)

    def carry_to(self, x_E: float, y_E: float, yaw_E: float) -> G1Waypoint:
        """Walk with the load to an aisle waypoint, allowing for the G1's steady walking offset."""
        return G1Waypoint(
            (x_E, y_E, yaw_E), ARMS_CARRY, tolerance_m=0.14, tolerance_rad=0.15, arm_time_s=0.1, max_steps=800
        )

    def unload_at(self, x_E: float, y_E: float, yaw_E: float, retreat_xy_E: tuple[float, float]) -> list[G1Waypoint]:
        """Lower the carton onto its support, release it and step clear."""
        target = (x_E, y_E, yaw_E)
        return [
            G1Waypoint(target, ARMS_CARRY, tolerance_m=0.11, tolerance_rad=0.15, max_steps=650, hold_steps=20),
            G1Waypoint(target, ARMS_PLACE, tolerance_m=0.13, tolerance_rad=0.15, max_steps=120, hold_steps=20),
            G1Waypoint(
                target, ARMS_WIDE, tolerance_m=0.13, tolerance_rad=0.15, arm_time_s=2.5, max_steps=170, hold_steps=20
            ),
            G1Waypoint((*retreat_xy_E, yaw_E), ARMS_WIDE),
        ]

    def inspection_and_pickup(self) -> list[G1Waypoint]:
        """Inspect the bays and pick up the carton on the common loading path."""
        return [
            G1Waypoint(
                (-0.12, 0, math.pi - 0.75),
                ARMS_WIDE,
                tolerance_m=0.16,
                tolerance_rad=0.18,
                hold_steps=15,
                max_steps=500,
            ),
            G1Waypoint(
                (-0.12, 0, -math.pi + 0.75),
                ARMS_WIDE,
                tolerance_m=0.16,
                tolerance_rad=0.18,
                hold_steps=15,
                max_steps=500,
            ),
            *self.approach_and_pickup(),
        ]

    def approach_and_pickup(self) -> list[G1Waypoint]:
        """Approach the loading table facing +x, lift the carton and step clear."""
        return [
            G1Waypoint((-0.25, 0, 0), ARMS_WIDE, tolerance_m=0.15, max_steps=650),
            G1Waypoint((0, 0, 0), ARMS_WIDE, tolerance_m=0.08, hold_steps=12, max_steps=200),
            G1Waypoint((0, 0, 0), ARMS_SQUEEZE, tolerance_m=0.10, hold_steps=15, max_steps=125),
            G1Waypoint((0, 0, 0), ARMS_CARRY, tolerance_m=0.10, hold_steps=20, max_steps=125),
            G1Waypoint((-0.4, 0, 0), ARMS_CARRY, tolerance_m=0.09, arm_time_s=0.1, max_steps=250),
        ]
