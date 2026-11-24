# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deliver a carton to the room bearing a color queried after the room markers leave view."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose
from isaaclab_arena_environments.spatial_memory.parts import ITEM_COLORS, board
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample
from isaaclab_arena_environments.spatial_memory.tasks.bay_delivery import SHELF_X_M, BayDeliveryLogic
from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LANE_PITCH_M


class PartitionDeliveryLogic(BayDeliveryLogic):
    """Inspect four marked rooms and enter the one named at the loading table."""

    name = "spatial_memory_partition_delivery"
    task_description = "Deliver the carton to the room matching the requested color."

    def obstacle_names(self):
        return list(ITEM_COLORS[:4])

    def build_part_specs(self):
        fixtures = self.receiving_parts()
        fixtures.extend(board(name, (0.06, 0.21, 0.24), name) for name in self.obstacle_names())
        for i in range(5):
            fixtures.append(board(f"wall_{i}", (0.84, 0.05, 1.45), "gray"))
        return fixtures

    def sample_layout(self, generator):
        poses, centers, offset, _ = self._bay_layout(generator)
        rooms = torch.randperm(4, generator=generator)
        query = torch.randint(4, (), generator=generator)
        for i, name in enumerate(self.obstacle_names()):
            # Each plaque overlaps the end of the left jamb and extends into its own doorway.
            jamb_y = float(centers[rooms[i]]) - LANE_PITCH_M / 2
            poses[name] = pose(SHELF_X_M + 0.61, jamb_y + 0.075, -0.12)
        for i in range(5):
            poses[f"wall_{i}"] = pose(SHELF_X_M + 0.18, offset + (i - 2) * LANE_PITCH_M, G1_FLOOR_Z_M + 0.725)
        return LayoutSample(
            self.part_poses_tensor(poses),
            {
                "t_E_start_xy": 0.04 * torch.rand(2, generator=generator) - 0.02,
                "t_E_target_y": centers[rooms[query]],
                "color": query,
            },
        )

    def turn_toward_bay(self, y_E):
        """Step clear of the jambs before swinging the carton around."""
        return [self.carry_to(-0.48, y_E, 0), self.carry_to(-0.48, y_E, math.pi)]

    def instruction_dim(self):
        return 4

    def instruction(self, env, episode):
        color = torch.nn.functional.one_hot(episode.layout_params["color"], self.instruction_dim())
        return color.float() * episode.instruction_revealed[:, None]
