# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Box-shaped placement asset for the relation-solver example notebooks."""

from __future__ import annotations

from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class ExampleObject(PlaceableAsset):
    """Box-shaped placement asset for the relation-solver example notebooks, with no Isaac Sim dependency."""

    def __init__(self, name: str, bounding_box: AxisAlignedBoundingBox):
        """Create a placement asset with a fixed bounding box.

        Args:
            name: Name of the asset.
            bounding_box: The local axis-aligned bounding box of the asset.
        """
        super().__init__(name=name)
        self._bounding_box = bounding_box

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the bounding box of the asset."""
        return self._bounding_box
