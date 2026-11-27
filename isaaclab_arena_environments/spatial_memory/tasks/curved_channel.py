# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace the centerline of a previously inspected curved rack while backing out."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.delivery_aisle import DeliveryAisleLogic


class CurvedChannelLogic(DeliveryAisleLogic):
    """Back through a curved channel and clear its end with the carton."""

    name = "spatial_memory_curved_channel"
    task_description = "Survey the bend and carry the carton backwards along the curved rack."
    episode_length_s = 130.0
    t_E_dispatch_xy = (-4.58, 1.65)
    minimum_env_spacing_m = 10.0
    stations_x_E = (-0.95, -1.65, -2.35, -3.05, -3.75)
    """Successive centerline stations between the common inlet and the hidden outlet."""

    def obstacle_names(self):
        return [f"{side}_{i}" for side in ("left", "right") for i in range(4)]

    def build_part_specs(self):
        parts = self.pickup_and_dispatch_parts()
        parts.extend(board(name, (0.90, 0.09, 0.67), "gray") for name in self.obstacle_names())
        return parts

    def sample_layout(self, generator):
        signs = 2 * torch.randint(2, (2,), generator=generator) - 1
        first_bend = float(signs[0]) * uniform(generator, 0.42, 0.57)
        second_bend = float(signs[1]) * uniform(generator, 0.45, 0.62)
        stations_y = torch.tensor(
            [0, first_bend * 0.32, first_bend, first_bend + second_bend * 0.38, first_bend + second_bend]
        )
        poses = {}
        for i in range(4):
            x, next_x = self.stations_x_E[i : i + 2]
            y, next_y = stations_y[i : i + 2].tolist()
            heading = math.atan2(next_y - y, next_x - x)
            for side, sign in (("left", -1), ("right", 1)):
                poses[f"{side}_{i}"] = pose(
                    (x + next_x) / 2, (y + next_y) / 2 + sign * 0.53, G1_FLOOR_Z_M + 0.335, heading
                )
        return self.aisle_layout(
            poses,
            {
                "t_E_centerline_y": stations_y,
                "t_E_start_xy": 0.04 * torch.rand(2, generator=generator) - 0.02,
            },
        )

    def on_step(self, env, episode, visible, active):
        t_E_pelvis = self._pelvis_xy_e(env)
        surveyed = episode.visible_before_hidden_phase.all(dim=1)
        self.update_memory_probe(
            env,
            episode,
            visible,
            active,
            reveal_allowed=(t_E_pelvis.norm(dim=1) < 0.12) & surveyed & episode.gripper_closed,
        )
        for station in (2, 4):
            crossed = episode.task_state_buffer(f"passed_{station}", False, torch.bool)
            crossing = active & ~crossed & (t_E_pelvis[:, 0] <= self.stations_x_E[station])
            inside = (t_E_pelvis[:, 1] - episode.layout_params["t_E_centerline_y"][:, station]).abs() < 0.29
            missed = episode.task_state_buffer("missed_bend", False, torch.bool)
            missed |= crossing & ~(inside & episode.instruction_revealed & self._carrying(env))
            crossed |= crossing

    def commitment_made(self, env, episode, active):
        return episode.task_state_buffer("passed_4", False, torch.bool)

    def extra_failures(self, env, episode):
        return super().extra_failures(env, episode) | episode.task_state_buffer("missed_bend", False, torch.bool)

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        terms["cleared_channel"] = episode.task_state_buffer("passed_4", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        ys = (
            episode.layout_params["t_E_centerline_y"][env_id].tolist()
            if mode == PlanMode.ORACLE
            else [0, 0.16, 0.5, 0.30, -0.02]
        )
        points = [self.carry_to(x, y, 0) for x, y in zip(self.stations_x_E, ys, strict=True)]
        return [
            *self.inspection_and_pickup(),
            *points,
            self.carry_to(-4.58, ys[-1], 0),
            self.carry_to(-4.58, ys[-1], math.pi / 2),
            *self.unload_at_dispatch(),
        ]
