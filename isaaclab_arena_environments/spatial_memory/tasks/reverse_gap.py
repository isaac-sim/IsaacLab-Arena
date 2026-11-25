# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect a loading barrier, then carry a box backwards through its remembered opening."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.box_carry import (
    LOAD_AHEAD_M,
    LOAD_CENTER_Z_M,
    TABLE_AHEAD_M,
    TABLE_SIZE_XY_M,
)
from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LANE_PITCH_M, LoadingLaneLogic

BARRIER_X_M = -1.15
"""Environment-frame longitudinal coordinate of the transverse loading barrier."""


RECEIVING_Y_M = 1.84
"""Pelvis y at placement, beyond the outermost passage."""


class ReverseGapLogic(LoadingLaneLogic):
    """Back through the remembered gap and deliver the box beyond the barrier."""

    name = "spatial_memory_reverse_gap"
    task_description = "Inspect the opening, then back through it carrying the box to the receiving table."
    episode_length_s = 100.0

    def obstacle_names(self):
        return ["barrier_left", "barrier_middle", "barrier_right"]

    def build_part_specs(self):
        table_size = (*TABLE_SIZE_XY_M, self.table_height())
        parts = self.pickup_parts()
        parts.append(board("dest_table", table_size))
        parts.extend(board(name, (0.18, LANE_PITCH_M, 0.73), "gray") for name in self.obstacle_names())
        return parts

    def sample_layout(self, generator):
        gap = int(torch.randint(4, (), generator=generator))
        offset_y = uniform(generator, -0.06, 0.06)
        table_z = G1_FLOOR_Z_M + self.table_height() / 2
        poses = {
            "pick_table": pose(TABLE_AHEAD_M, 0, table_z),
            "load": pose(LOAD_AHEAD_M, 0, LOAD_CENTER_Z_M + 0.002),
            "dest_table": pose(-2.15, RECEIVING_Y_M + TABLE_AHEAD_M, table_z, math.pi / 2),
        }
        lane_centers = (torch.arange(4) - 1.5) * LANE_PITCH_M + offset_y
        blocks = iter(self.obstacle_names())
        for lane, center in enumerate(lane_centers):
            if lane != gap:
                poses[next(blocks)] = pose(BARRIER_X_M, float(center), G1_FLOOR_Z_M + 0.365)
        params = {
            "t_E_target_y": lane_centers[gap],
            "t_E_start_xy": 0.04 * torch.rand(2, generator=generator) - 0.02,
            "dest_table_xy": torch.tensor([-2.15, RECEIVING_Y_M + TABLE_AHEAD_M]),
            "dest_y": torch.tensor(RECEIVING_Y_M),
        }
        return LayoutSample(self.part_poses_tensor(poses), params)

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        terms["correct_lane"] = episode.task_state_buffer("correct_lane", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        gap_y = float(episode.layout_params["t_E_target_y"][env_id]) if mode == PlanMode.ORACLE else -0.5 * LANE_PITCH_M
        return [*self.inspection_and_pickup(), *self.deliver_to_lane(gap_y)]

    def deliver_to_lane(self, lane_y_E: float):
        """Cross the selected lane and unload the carton using its supplied lateral position."""
        return [
            self.carry_to(-0.4, lane_y_E, 0),
            self.carry_to(-1.91, lane_y_E, 0),
            self.carry_to(-2.15, lane_y_E, math.pi / 2),
            *self.unload_at(-2.15, RECEIVING_Y_M, math.pi / 2, (-2.15, RECEIVING_Y_M - 0.45)),
        ]
