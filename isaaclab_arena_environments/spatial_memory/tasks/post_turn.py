# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Retreat past a post and turn the carton before the downstream stop."""

from __future__ import annotations

import math
import torch
from dataclasses import replace

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartShape, board
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.delivery_aisle import DeliveryAisleLogic


class PostTurnLogic(DeliveryAisleLogic):
    """Choose when to rotate a projecting load inside a finite turning window."""

    name = "spatial_memory_post_turn"
    task_description = "Inspect the post and stop, then back up and swing the carton into the dispatch aisle."
    t_E_dispatch_xy = (-1.95, 1.96)
    minimum_env_spacing_m = 8.0
    episode_length_s = 115.0

    def obstacle_names(self):
        return ["corner_post", "end_stop"]

    def build_part_specs(self):
        parts = self.pickup_and_dispatch_parts()
        parts.append(replace(board("corner_post", (0.26, 0.26, 1.0), "gray"), shape=PartShape.CYLINDER))
        parts.append(board("end_stop", (0.14, 1.12, 0.73), "gray"))
        return parts

    def sample_layout(self, generator):
        window = int(torch.randint(4, (), generator=generator))
        turn_x_E = -1.24 - window * 0.68 + uniform(generator, -0.04, 0.04)
        fixtures = {
            "corner_post": pose(turn_x_E + 0.35, 0.46, G1_FLOOR_Z_M + 0.5),
            "end_stop": pose(turn_x_E - 0.44, 0.11, G1_FLOOR_Z_M + 0.365),
        }
        return self.aisle_layout(
            fixtures,
            {
                "t_E_turn_x": torch.tensor(turn_x_E),
                "t_E_start_xy": 0.04 * torch.rand(2, generator=generator) - 0.02,
            },
        )

    def on_step(self, env, episode, visible, active):
        t_E_pelvis = self._pelvis_xy_e(env)
        at_pickup = t_E_pelvis.norm(dim=1) < 0.12
        self.update_memory_probe(
            env,
            episode,
            visible,
            active,
            reveal_allowed=at_pickup & episode.gripper_closed & episode.visible_before_hidden_phase.all(dim=1),
        )
        turned = self.commitment_made(env, episode, active)
        t_E_load = env.arena_world.get_pose_e("load")
        turning = active & ~turned & self._carrying(env) & (t_E_load[:, 1] - t_E_pelvis[:, 1] > 0.19)
        inside = (t_E_pelvis[:, 0] - episode.layout_params["t_E_turn_x"]).abs() < 0.29
        correct = episode.task_state_buffer("correct_lane", False, torch.bool)
        correct[turning] = (inside & episode.instruction_revealed)[turning]
        turned |= turning

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        terms["turned_between_post_and_stop"] = episode.task_state_buffer("correct_lane", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        turn_x_E = float(episode.layout_params["t_E_turn_x"][env_id]) if mode == PlanMode.ORACLE else -1.92
        return [
            *self.inspection_and_pickup(),
            self.carry_to(turn_x_E, 0, 0),
            self.carry_to(turn_x_E, 0, math.pi / 2),
            self.carry_to(turn_x_E, 1.45, math.pi / 2),
            self.carry_to(-1.95, 1.45, math.pi / 2),
            *self.unload_at_dispatch(),
        ]
