# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn a carried carton around inside the only full-depth recess."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.delivery_aisle import DeliveryAisleLogic


class TurningPocketLogic(DeliveryAisleLogic):
    """Enter the deep pocket, reverse heading with the carton and return to dispatch."""

    name = "spatial_memory_turning_pocket"
    task_description = "Inspect the recesses and use the deep one to turn the carton around before delivery."
    t_E_dispatch_xy = (0.05, 2.27)
    minimum_env_spacing_m = 8.0
    episode_length_s = 130.0

    def obstacle_names(self):
        return [f"back_{i}" for i in range(4)]

    def build_part_specs(self):
        parts = self.pickup_and_dispatch_parts()
        parts.extend(board(name, (0.12, 1.28, 0.80), "gray") for name in self.obstacle_names())
        for i in range(5):
            parts.append(board(f"divider_{i}", (1.56, 0.08, 0.80), "gray"))
        return parts

    def sample_layout(self, generator):
        deep = int(torch.randint(4, (), generator=generator))
        offset = 0.08 * float(torch.rand((), generator=generator)) - 0.04
        centers = (torch.arange(4) - 1.5) * 1.28 + offset
        poses = {}
        for i, name in enumerate(self.obstacle_names()):
            poses[name] = pose(-2.55 if i == deep else -1.42, float(centers[i]), G1_FLOOR_Z_M + 0.4)
        for i in range(5):
            poses[f"divider_{i}"] = pose(-1.80, (i - 2) * 1.28 + offset, G1_FLOOR_Z_M + 0.4)
        return self.aisle_layout(
            poses,
            {
                "t_E_target_y": centers[deep],
                "t_E_start_xy": 0.04 * torch.rand(2, generator=generator) - 0.02,
            },
        )

    def on_step(self, env, episode, visible, active):
        super().on_step(env, episode, visible, active)
        t_E_pelvis = self._pelvis_xy_e(env)
        t_E_carton = env.arena_world.get_pose_e("load")
        facing_outbound = t_E_carton[:, 0] < t_E_pelvis[:, 0] - 0.29
        inside = (t_E_pelvis[:, 0] < -1.55) & ((t_E_pelvis[:, 1] - episode.layout_params["t_E_target_y"]).abs() < 0.28)
        turned = episode.task_state_buffer("turned_inside", False, torch.bool)
        turned |= active & self._carrying(env) & inside & facing_outbound

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        terms["turned_inside"] = episode.task_state_buffer("turned_inside", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        pocket_y = float(episode.layout_params["t_E_target_y"][env_id]) if mode == PlanMode.ORACLE else -0.64
        return [
            *self.inspection_and_pickup(),
            self.carry_to(-0.40, pocket_y, 0),
            self.carry_to(-1.78, pocket_y, 0),
            self.carry_to(-1.78, pocket_y, math.pi / 2),
            self.carry_to(-1.78, pocket_y, math.pi),
            self.carry_to(-0.26, pocket_y, math.pi),
            self.carry_to(-0.26, 2.05, math.pi),
            self.carry_to(0.05, 2.05, math.pi / 2),
            *self.unload_at_dispatch(),
        ]
