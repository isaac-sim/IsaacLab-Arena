# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect a carton and back around the remembered end of a rack wall."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import ARMS_WIDE, G1_FLOOR_Z_M, G1Waypoint
from isaaclab_arena_environments.spatial_memory.geometry import FACE_NX, FACE_PX
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.delivery_aisle import DeliveryAisleLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

WALL_LENGTH_M = 1.80
"""Length of each wall along the withdrawal aisle."""
GAP_WIDTH_M = 1.25
"""Separation of the two rack ends along the aisle."""
WALL_Y_E = 0.90
"""Lateral coordinate of the divider between pickup and dispatch."""


class DoglegExitLogic(DeliveryAisleLogic):
    """Remember how far to retreat before crossing the rack divider with the carton."""

    name = "spatial_memory_dogleg_exit"
    t_E_dispatch_xy = (-1.55, 2.08)
    minimum_env_spacing_m = 8.0
    episode_length_s = 120.0
    task_description = "Inspect the rack ends, collect the carton and withdraw through the dogleg to dispatch."

    def obstacle_names(self):
        return ["rack_end", "end_stop"]

    def visibility_targets(self):
        return [
            VisibilityTarget("rack_end", "rack_end", (FACE_NX,), spacing_m=0.04),
            VisibilityTarget("end_stop", "end_stop", (FACE_PX,), spacing_m=0.04),
        ]

    def build_part_specs(self):
        parts = self.pickup_and_dispatch_parts()
        parts.append(board("pickup_screen", (2.80, 0.08, 1.75), "gray"))
        parts.extend(board(name, (WALL_LENGTH_M, 0.16, 0.75), "gray") for name in self.obstacle_names())
        return parts

    def sample_layout(self, generator):
        slot = int(torch.randint(4, (), generator=generator))
        gap_x = -1.05 - 0.64 * slot + uniform(generator, -0.04, 0.04)
        half_span = (WALL_LENGTH_M + GAP_WIDTH_M) / 2
        poses = {
            "pickup_screen": pose(0.88, 0.63, G1_FLOOR_Z_M + 0.875),
            "rack_end": pose(gap_x + half_span, WALL_Y_E, G1_FLOOR_Z_M + 0.375),
            "end_stop": pose(gap_x - half_span, WALL_Y_E, G1_FLOOR_Z_M + 0.375),
        }
        start_xy = torch.tensor([-3.08, -0.06]) + 0.04 * torch.rand(2, generator=generator)
        params = {
            "gap_x_E": torch.tensor(gap_x),
            "t_E_start_xy": start_xy,
        }
        return self.aisle_layout(poses, params)

    def on_step(self, env, episode, visible, active):
        xy_E = self._pelvis_xy_e(env)
        surveyed = episode.visible_before_hidden_phase.all(dim=1)
        self.update_memory_probe(
            env, episode, visible, active, reveal_allowed=(xy_E.norm(dim=1) < 0.12) & surveyed & episode.gripper_closed
        )
        crossed = self.commitment_made(env, episode, active)
        crossing = active & ~crossed & (xy_E[:, 1] > 0.37)
        correct = episode.task_state_buffer("correct_lane", False, torch.bool)
        aligned = (xy_E[:, 0] - episode.layout_params["gap_x_E"]).abs() < 0.31
        correct[crossing] = (aligned & episode.instruction_revealed & self._carrying(env))[crossing]
        crossed |= crossing

    def success_terms(self, env, episode):
        terms = super().success_terms(env, episode)
        terms["through_dogleg"] = episode.task_state_buffer("correct_lane", False, torch.bool)
        return terms

    def plan(self, env, episode, env_id, mode):
        gap_x = float(episode.layout_params["gap_x_E"][env_id]) if mode == PlanMode.ORACLE else -1.69
        survey = [
            G1Waypoint((x, 0, math.pi / 2), ARMS_WIDE, tolerance_m=0.16, tolerance_rad=0.18, max_steps=850)
            for x in (-3.0, -1.9, -0.85)
        ]
        return [
            *survey,
            *self.approach_and_pickup(),
            self.carry_to(-0.48, 0, -math.pi / 2),
            self.carry_to(gap_x, 0, -math.pi / 2),
            self.carry_to(gap_x, 1.86, -math.pi / 2),
            self.carry_to(-1.55, 1.84, math.pi),
            self.carry_to(-1.55, 1.84, math.pi / 2),
            *self.unload_at_dispatch(),
        ]
