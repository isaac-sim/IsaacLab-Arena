# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect a four-bay shelf, then carry a carton to the bay remembered as empty."""

from __future__ import annotations

import torch

from isaaclab_arena_environments.spatial_memory.layout_sampling import pose
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample
from isaaclab_arena_environments.spatial_memory.tasks.bay_delivery import SHELF_X_M, BayDeliveryLogic
from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LANE_PITCH_M


class ShelfBayLogic(BayDeliveryLogic):
    """Choose a shelf bay while facing the loading table, then turn and insert the carton."""

    name = "spatial_memory_shelf_bay"
    task_description = "Remember the empty shelf bay and restock it with the carton."

    def obstacle_names(self):
        return ["stock_a", "stock_b", "stock_c"]

    def build_part_specs(self):
        parts = self.receiving_parts()
        parts.append(board("shelf_top", (0.52, 4 * LANE_PITCH_M, 0.06)))
        for i in range(5):
            parts.append(board(f"divider_{i}", (0.50, 0.045, 0.60)))
        parts.extend(board(name, (0.28, 0.52, 0.32), "blue") for name in self.obstacle_names())
        return parts

    def sample_layout(self, generator):
        poses, centers, offset, top_z = self._bay_layout(generator)
        empty = int(torch.randint(4, (), generator=generator))
        for i in range(5):
            poses[f"divider_{i}"] = pose(SHELF_X_M, offset + (i - 2) * LANE_PITCH_M, top_z + 0.30)
        poses["shelf_top"] = pose(SHELF_X_M, offset, top_z + 0.60)
        stock = iter(self.obstacle_names())
        for i, y in enumerate(centers):
            if i != empty:
                poses[next(stock)] = pose(SHELF_X_M, float(y), top_z + 0.16)
        return LayoutSample(
            self.part_poses_tensor(poses),
            {"t_E_start_xy": 0.04 * torch.rand(2, generator=generator) - 0.02, "t_E_target_y": centers[empty]},
        )
