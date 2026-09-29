# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams


@dataclass
class PlacementRecordingParams(SettledPlacementParams):
    """Physics duration, acceptance checks and minimum recording yield."""

    min_layouts: int = 1
    """Minimum accepted layouts required before writing the recording."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.min_layouts > 0, "min_layouts must be positive"
