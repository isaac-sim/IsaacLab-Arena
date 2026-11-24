# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 loading stations with four adjacent receiving bays."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.predicates import part_upright
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.box_carry import LOAD_AHEAD_M, LOAD_CENTER_Z_M, TABLE_AHEAD_M
from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LANE_PITCH_M, LoadingLaneLogic

SHELF_X_M = -1.85
"""Environment-frame x coordinate of the shelf center."""


class BayDeliveryLogic(LoadingLaneLogic):
    """Deliver a carton to a remembered bay on the receiving bench."""

    episode_length_s = 105.0

    def receiving_parts(self):
        """Build the common loading and receiving tables and carton."""
        parts = self.pickup_parts()
        parts.append(board("dest_table", (0.50, 4 * LANE_PITCH_M, self.table_height())))
        return parts

    def _bay_layout(self, generator: torch.Generator):
        """Sample the bench pose and return its bay centers, lateral offset and top height."""
        offset = uniform(generator, -0.06, 0.06)
        centers = (torch.arange(4) - 1.5) * LANE_PITCH_M + offset
        top_z = G1_FLOOR_Z_M + self.table_height()
        table_z = G1_FLOOR_Z_M + self.table_height() / 2
        poses = {
            "pick_table": pose(TABLE_AHEAD_M, 0, table_z),
            "load": pose(LOAD_AHEAD_M, 0, LOAD_CENTER_Z_M + 0.002),
            "dest_table": pose(SHELF_X_M, offset, table_z),
        }
        return poses, centers, offset, top_z

    def turn_toward_bay(self, y_E):
        """Turn the carried carton toward the selected bay after crossing the approach line."""
        return [self.carry_to(-0.93, y_E, math.pi)]

    def plan(self, env, episode, env_id, mode):
        y = float(episode.layout_params["t_E_target_y"][env_id]) if mode == PlanMode.ORACLE else -0.5 * LANE_PITCH_M
        return [
            *self.inspection_and_pickup(),
            self.carry_to(-0.4, y, 0),
            self.carry_to(-0.93, y, 0),
            *self.turn_toward_bay(y),
            *self.unload_at(SHELF_X_M + TABLE_AHEAD_M, y, math.pi, (-0.91, y)),
        ]

    def success_terms(self, env, episode):
        load = env.arena_world.get_pose_e("load")
        return {
            "in_bay": (load[:, 0] - SHELF_X_M).abs() < 0.22,
            "at_remembered_slot": (load[:, 1] - episode.layout_params["t_E_target_y"]).abs() < 0.30,
            "supported": (load[:, 2] - LOAD_CENTER_Z_M).abs() < 0.03,
            "upright": part_upright(env, "load"),
            "released": ~self.gripper_closed(env) & (self._pelvis_xy_e(env)[:, 0] > -1.10),
            "correct_lane": episode.task_state_buffer("correct_lane", False, torch.bool),
        }
