# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Find the continuous passage through two independently blocked loading gates."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import ARMS_WIDE, G1_FLOOR_Z_M, G1Waypoint
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.delivery_aisle import DeliveryAisleLogic
from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LANE_PITCH_M


class TwoViewExitLogic(DeliveryAisleLogic):
    """Carry the load through the lane that is clear across both gate rows."""

    name = "spatial_memory_two_view_exit"
    task_description = "Inspect both gates and back through the lane that is clear at both depths."
    t_E_dispatch_xy = (-13.0, 1.78)
    minimum_env_spacing_m = 20.0
    episode_length_s = 210.0

    def inside_lane_entrance(self, t_E_pelvis_xy):
        """Exclude the pedestrian gallery alongside the loading lanes."""
        return super().inside_lane_entrance(t_E_pelvis_xy) & (t_E_pelvis_xy[:, 1].abs() < 1.90)

    def obstacle_names(self):
        return ["outer_left", "outer_right", "inner_left", "inner_right"]

    def build_part_specs(self):
        parts = self.pickup_and_dispatch_parts()
        parts.extend(board(name, (0.14, LANE_PITCH_M, 0.68), "gray") for name in self.obstacle_names())
        parts.append(board("gallery_screen", (6.4, 0.10, 1.60), "gray"))
        for i in range(3):
            parts.append(board(f"rail_{i}", (10.85, 0.06, 0.44), "gray"))
        return parts

    def sample_layout(self, generator):
        half = int(torch.randint(2, (), generator=generator))
        parity = int(torch.randint(2, (), generator=generator))
        offset = uniform(generator, -0.05, 0.05)
        centers = (torch.arange(4) - 1.5) * LANE_PITCH_M + offset
        poses = {"gallery_screen": pose(-6.575, -2.06, G1_FLOOR_Z_M + 0.80)}
        for index in range(2):
            outer_lane = 2 * (1 - half) + index
            inner_lane = 2 * index + (1 - parity)
            poses[self.obstacle_names()[index]] = pose(-1.15, float(centers[outer_lane]), G1_FLOOR_Z_M + 0.34)
            poses[self.obstacle_names()[index + 2]] = pose(-12.0, float(centers[inner_lane]), G1_FLOOR_Z_M + 0.34)
        for index in range(3):
            poses[f"rail_{index}"] = pose(-6.575, offset + (index - 1) * LANE_PITCH_M, G1_FLOOR_Z_M + 0.22)
        return self.aisle_layout(
            poses,
            {
                "t_E_target_y": centers[2 * half + parity],
                "t_E_start_xy": torch.tensor([-10.42, -2.63]) + 0.04 * torch.rand(2, generator=generator),
            },
        )

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        terms["correct_lane"] = episode.task_state_buffer("correct_lane", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        lane_y = (
            float(episode.layout_params["t_E_target_y"][env_id]) if mode == PlanMode.ORACLE else -0.5 * LANE_PITCH_M
        )
        return [
            G1Waypoint((-10.4, -2.61, 2.56), ARMS_WIDE, hold_steps=20),
            G1Waypoint((-10.4, -2.61, 1.91), ARMS_WIDE, hold_steps=20),
            G1Waypoint((-9.7, -2.65, 0), ARMS_WIDE, max_steps=650),
            G1Waypoint((-0.10, -2.65, 0), ARMS_WIDE, max_steps=1850),
            G1Waypoint((-0.10, -1.40, math.pi - 0.62), ARMS_WIDE, max_steps=700),
            *self.inspection_and_pickup(),
            self.carry_to(-0.4, lane_y, 0),
            *[self.carry_to(x_E, lane_y, 0) for x_E in (-3, -6, -9, -11, -13)],
            self.carry_to(-13, lane_y, math.pi / 2),
            *self.unload_at_dispatch(),
        ]
