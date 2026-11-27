# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pickup and dispatch geometry for carrying a carton through a storage aisle."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample
from isaaclab_arena_environments.spatial_memory.tasks.box_carry import (
    LOAD_AHEAD_M,
    LOAD_CENTER_Z_M,
    TABLE_AHEAD_M,
    TABLE_SIZE_XY_M,
)
from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LoadingLaneLogic


class DeliveryAisleLogic(LoadingLaneLogic):
    """Pick up facing +x and unload at a fixed dispatch table facing +y."""

    t_E_dispatch_xy: tuple[float, float]
    """Pelvis stance in the environment frame when unloading at dispatch."""

    def pickup_and_dispatch_parts(self):
        """Build the carton and both supporting tables."""
        return [*self.pickup_parts(), board("dest_table", (*TABLE_SIZE_XY_M, self.table_height()))]

    def aisle_layout(self, fixtures, params):
        """Add the loading station and dispatch table to the sampled aisle fixtures."""
        x_E, y_E = self.t_E_dispatch_xy
        dispatch_xy_E = (x_E, y_E + TABLE_AHEAD_M)
        table_z_E = G1_FLOOR_Z_M + self.table_height() / 2
        poses = {
            **fixtures,
            "pick_table": pose(TABLE_AHEAD_M, 0, table_z_E),
            "load": pose(LOAD_AHEAD_M, 0, LOAD_CENTER_Z_M + 0.002),
            "dest_table": pose(*dispatch_xy_E, table_z_E, math.pi / 2),
        }
        return LayoutSample(
            self.part_poses_tensor(poses),
            {
                **params,
                "dest_table_xy": torch.tensor(dispatch_xy_E),
                "dest_y": torch.tensor(y_E),
            },
        )

    def unload_at_dispatch(self):
        """Place the carton at dispatch and step away from the support."""
        x_E, y_E = self.t_E_dispatch_xy
        return self.unload_at(x_E, y_E, math.pi / 2, (x_E, y_E - 0.46))
