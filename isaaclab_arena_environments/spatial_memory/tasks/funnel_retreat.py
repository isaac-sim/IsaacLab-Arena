# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reverse down a tapering pair of rails whose outlet is hidden at pickup."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.delivery_aisle import DeliveryAisleLogic

THROAT_X_E = -2.65
"""Longitudinal position at which the rails are closest."""


class FunnelRetreatLogic(DeliveryAisleLogic):
    """Set a loaded diagonal from the loading table to the remembered funnel throat."""

    name = "spatial_memory_funnel_retreat"
    t_E_dispatch_xy = (-3.56, 1.72)
    minimum_env_spacing_m = 8.0
    task_description = "Survey the funnel and reverse through its narrow end carrying the carton."
    episode_length_s = 110.0

    def obstacle_names(self):
        return ["left_jaw", "right_jaw"]

    def build_part_specs(self):
        parts = self.pickup_and_dispatch_parts()
        parts.extend(board(name, (1.62, 0.10, 0.72), "gray") for name in self.obstacle_names())
        return parts

    def sample_layout(self, generator):
        index = int(torch.randint(4, (), generator=generator))
        throat_y = (index - 1.5) * 0.82 + uniform(generator, -0.04, 0.04)
        route_yaw = math.atan2(throat_y, 2.25)
        poses = {}
        for name, side in zip(self.obstacle_names(), (-1, 1), strict=True):
            # Pivot each rail about its outlet end, with the inlet wider on both sides.
            yaw = -route_yaw + side * 0.20
            poses[name] = pose(
                THROAT_X_E + 0.81 * math.cos(yaw),
                throat_y + side * 0.52 + 0.81 * math.sin(yaw),
                G1_FLOOR_Z_M + 0.36,
                yaw,
            )
        return self.aisle_layout(
            poses,
            {
                "t_E_throat_y": torch.tensor(throat_y),
                "t_E_start_xy": 0.04 * torch.rand(2, generator=generator) - 0.02,
            },
        )

    def on_step(self, env, episode, visible, active):
        t_E_pelvis = self._pelvis_xy_e(env)
        ready = (t_E_pelvis.norm(dim=1) < 0.12) & episode.visible_before_hidden_phase.all(dim=1)
        self.update_memory_probe(env, episode, visible, active, reveal_allowed=ready & episode.gripper_closed)
        committed = self.commitment_made(env, episode, active)
        passing = active & ~committed & (t_E_pelvis[:, 0] <= THROAT_X_E)
        aligned = (t_E_pelvis[:, 1] - episode.layout_params["t_E_throat_y"]).abs() < 0.32
        correct = episode.task_state_buffer("correct_lane", False, torch.bool)
        correct[passing] = (aligned & episode.instruction_revealed & self._carrying(env))[passing]
        committed |= passing

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        terms["through_throat"] = episode.task_state_buffer("correct_lane", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        throat_y = float(episode.layout_params["t_E_throat_y"][env_id]) if mode == PlanMode.ORACLE else -0.41
        return [
            *self.inspection_and_pickup(),
            self.carry_to(-1.52, throat_y / 2, 0),
            self.carry_to(THROAT_X_E, throat_y, 0),
            self.carry_to(-3.56, throat_y, 0),
            self.carry_to(-3.56, throat_y, math.pi / 2),
            *self.unload_at_dispatch(),
        ]
