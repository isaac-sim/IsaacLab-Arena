# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class InitializerType(Enum):
    """Available placement initialization strategies."""

    ANCHOR = "anchor"
    """Seed against the footprint of the first anchor at or above each object's On parent."""


class PlacementInitializerBase(ABC):
    """Produces an initialization for the relation solver.

    Generates positions per object per env that the relation solver optimizes from.
    """

    @abstractmethod
    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, tuple[float, float, float]]:
        """Return one starting position per object for a single solver candidate.

        Args:
            objects: Every object taking part in the solve, anchors included.
            anchor_objects: The subset of objects that stay at their fixed initial pose.
            asset_to_bbox: Local bounding box per object for the env this candidate belongs to.
                Each bounding box holds a single row, so min_point/max_point are shape (1, 3).
            generator: RNG for reproducible sampling. None uses PyTorch's global RNG.

        Returns:
            Position-initialization per object.
        """
