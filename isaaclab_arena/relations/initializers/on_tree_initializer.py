# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.initializers.initialization_bounds import get_initialization_bounds
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


class OnTreeInitializer(PlacementInitializerBase):
    """Generates initial positions for objects by walking down the ``On`` tree in the relation graph.

    Each object is sampled inside the footprint its own parent was just sampled at, narrowed by
    any of the object's other relations that constrain its position.
    """

    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox],
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, tuple[float, float, float]]:
        # Getting the first anchor's center, which will act as a fallback.
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        fallback_center = get_world_bbox_at_initial_pose(first_anchor, asset_to_bbox).center[0]
        fallback_position = (float(fallback_center[0]), float(fallback_center[1]), float(fallback_center[2]))

        positions: dict[PlaceableAsset, tuple[float, float, float]] = {}
        # Bounding boxes of the objects sampled so far, in the world frame, i.e. their local
        # bounding boxes translated by the position each one was just sampled at.
        sampled_world_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox] = {}
        ordered_objects: list[PlaceableAsset] = _order_parents_before_children(objects, anchor_objects)
        for obj in ordered_objects:
            if obj in anchor_objects:
                positions[obj] = get_fixed_anchor_position(obj)
            else:
                on_relation: On | None = get_relation(obj, On)
                if on_relation is None:
                    positions[obj] = fallback_position
                else:
                    parent_world_bbox: AxisAlignedBoundingBox = sampled_world_bboxes[on_relation.parent]
                    # Narrowing keeps a parent close to where it is sampled, so its children are
                    # not left behind when the solve pulls it towards its own constraints.
                    positions[obj] = sample_on_parent(
                        obj,
                        parent_world_bbox,
                        asset_to_bbox,
                        generator,
                        narrowing_bounds=get_initialization_bounds(obj),
                    )
            sampled_world_bboxes[obj] = asset_to_bbox[obj].translated(positions[obj])
        # Return the positions in the order the caller supplied the objects, rather than in the
        # parents-first order this method sampled them in.
        return {obj: positions[obj] for obj in objects}


def _order_parents_before_children(
    objects: list[PlaceableAsset],
    anchor_objects: set[PlaceableAsset],
) -> list[PlaceableAsset]:
    """Return objects ordered so every ``On`` parent precedes its children.

    Sampling a child needs its parent's sampled position, so the objects have to be visited in
    dependency order. The ordering is built in rounds, like a topological sort: objects that need
    no parent are placed first, then every round appends the objects whose parent is already
    ordered, until nothing is left.
    """
    # Anchors are at fixed poses and unparented objects fall back to the anchor center, so
    # neither needs a parent sampled first. These seed the ordering.
    ordered: list[PlaceableAsset] = []
    ordered_set: set[PlaceableAsset] = set()
    remaining: list[PlaceableAsset] = []
    for obj in objects:
        if obj in anchor_objects or get_relation(obj, On) is None:
            ordered.append(obj)
            ordered_set.add(obj)
        else:
            remaining.append(obj)

    while remaining:
        # An object is ready once its parent has been ordered. Each round therefore descends one
        # more level down the On tree.
        ready = [obj for obj in remaining if get_relation(obj, On).parent in ordered_set]
        # No ready object means every remaining object is waiting on a parent that will never be
        # ordered, which is a cycle or a parent outside the placement set.
        assert ready, (
            "On relations must form a forest rooted at anchors, but no parent could be resolved for "
            f"{[obj.name for obj in remaining]}. Check for a cycle or a parent outside the placement set."
        )
        ordered += ready
        ordered_set.update(ready)
        remaining = [obj for obj in remaining if obj not in ordered_set]
    return ordered
