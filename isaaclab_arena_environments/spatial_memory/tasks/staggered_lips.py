# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Withdraw through two shelf openings with independently projecting lips."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.box_carry import LOAD_CENTER_Z_M
from isaaclab_arena_environments.spatial_memory.tasks.delivery_aisle import DeliveryAisleLogic

LIP_X_E = (-1.15, -2.95)
"""Longitudinal positions of the two shelf edges along the exit."""


class StaggeredLipsLogic(DeliveryAisleLogic):
    """Require the carton and pelvis to clear both projections on their open sides."""

    name = "spatial_memory_staggered_lips"
    task_description = "Inspect both projecting shelf edges and carry the carton back along their open sides."
    t_E_dispatch_xy = (-3.87, 1.48)
    minimum_env_spacing_m = 9.0
    episode_length_s = 120.0

    def obstacle_names(self):
        return ["near_lip", "far_lip"]

    def build_part_specs(self):
        parts = self.pickup_and_dispatch_parts()
        parts.extend(board(name, (0.20, 0.98, 0.20)) for name in self.obstacle_names())
        for side in ("left", "right"):
            parts.append(board(f"upright_{side}", (2.28, 0.09, 1.22), "gray"))
        return parts

    def sample_layout(self, generator):
        poses = {}
        offset = uniform(generator, -0.04, 0.04)
        open_y = (2 * torch.randint(2, (2,), generator=generator) - 1) * 0.52
        for i, name in enumerate(self.obstacle_names()):
            poses[name] = pose(LIP_X_E[i], offset - float(open_y[i]), LOAD_CENTER_Z_M + 0.04)
        for side, sign in (("left", -1), ("right", 1)):
            poses[f"upright_{side}"] = pose(-2.05, offset + sign * 1.03, G1_FLOOR_Z_M + 0.61)
        return self.aisle_layout(
            poses,
            {
                "t_E_open_y": open_y + offset,
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
        for i, x_E in enumerate(LIP_X_E):
            crossed = episode.task_state_buffer(f"crossed_{i}", False, torch.bool)
            crossing = active & ~crossed & (t_E_pelvis[:, 0] < x_E)
            at_opening = (t_E_pelvis[:, 1] - episode.layout_params["t_E_open_y"][:, i]).abs() < 0.32
            correct = episode.task_state_buffer(f"cleared_{i}", False, torch.bool)
            correct[crossing] = (at_opening & episode.instruction_revealed & self._carrying(env))[crossing]
            crossed |= crossing

    def commitment_made(self, env, episode, active):
        return episode.task_state_buffer("crossed_1", False, torch.bool)

    def extra_failures(self, env, episode):
        failed = super().extra_failures(env, episode)
        for i in range(2):
            failed |= episode.task_state_buffer(f"crossed_{i}", False, torch.bool) & ~episode.task_state_buffer(
                f"cleared_{i}", False, torch.bool
            )
        return failed

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        for i in range(2):
            terms[f"cleared_{i}"] = episode.task_state_buffer(f"cleared_{i}", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        near_y, far_y = (
            episode.layout_params["t_E_open_y"][env_id].tolist() if mode == PlanMode.ORACLE else (-0.52, 0.52)
        )
        return [
            *self.inspection_and_pickup(),
            self.carry_to(-0.4, near_y, 0),
            self.carry_to(-2.06, near_y, 0),
            self.carry_to(-2.06, far_y, 0),
            self.carry_to(-3.87, far_y, 0),
            self.carry_to(-3.87, far_y, math.pi / 2),
            *self.unload_at_dispatch(),
        ]
