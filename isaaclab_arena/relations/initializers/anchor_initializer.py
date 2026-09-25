# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.initializers.placement_initializer_base import (
    PlacementInitializerBase,
    get_fixed_anchor_position,
    get_world_bbox_at_initial_pose,
    sample_on_parent,
)
from isaaclab_arena.relations.relations import On, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class AnchorInitializer(PlacementInitializerBase):
    """Seeds every object against an anchor's footprint.

    Objects with an ``On`` relation are sampled inside the footprint of the first anchor found by
    walking up their ``On`` chain. All other objects, and objects whose chain reaches no anchor,
    start at the first anchor's center.
    """

    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, tuple[float, float, float]]:
        # Getting the first anchor's bounding box, which will act as a fallback.
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        first_anchor_bbox = get_world_bbox_at_initial_pose(first_anchor, asset_to_bbox)
        center = first_anchor_bbox.center[0]
        first_anchor_center = (float(center[0]), float(center[1]), float(center[2]))

        positions: dict[PlaceableAsset, tuple[float, float, float]] = {}
        for obj in objects:
            if obj in anchor_objects:
                positions[obj] = get_fixed_anchor_position(obj)
            elif get_relation(obj, On) is not None:
                parent_bbox = self._get_first_anchor_bbox_above(obj, anchor_objects, first_anchor_bbox, asset_to_bbox)
                positions[obj] = sample_on_parent(obj, parent_bbox, asset_to_bbox, generator)
            else:
                positions[obj] = first_anchor_center
        return positions

    @staticmethod
    def _get_first_anchor_bbox_above(
        obj: PlaceableAsset,
        anchor_objects: set[PlaceableAsset],
        fallback_bbox: AxisAlignedBoundingBox,
        asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox],
    ) -> AxisAlignedBoundingBox:
        """Return the world bbox of the nearest anchor above obj in its ``On`` chain.

        Walks up the chain of ``On`` parents until it reaches an anchor. Chains that end without
        an anchor, and chains that loop, fall back to fallback_bbox.
        """
        visited: set[PlaceableAsset] = set()
        parent = get_relation(obj, On).parent
        while parent is not None and parent not in visited:
            if parent in anchor_objects:
                return get_world_bbox_at_initial_pose(parent, asset_to_bbox)
            visited.add(parent)
            parent_on_relation = get_relation(parent, On)
            parent = parent_on_relation.parent if parent_on_relation is not None else None
        return fallback_bbox
