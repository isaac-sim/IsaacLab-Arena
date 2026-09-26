# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.initializers.placement_initializer_base import (
    PlacementInitializerBase,
    get_fixed_anchor_position,
    get_on_parent_position_bbox,
    get_world_bbox_at_initial_pose,
    sample_position_in_bbox,
)
from isaaclab_arena.relations.relations import On, PositionLimitsBox, RelationBase, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class OnTreeInitializer(PlacementInitializerBase):
    """Generates initial positions for objects by walking down the ``On`` tree in the relation graph.

    Each object is sampled inside the footprint its own parent was just sampled at, narrowed by
    any of the object's other relations that constrain its position. Seeding a child inside its
    parent's footprint is only useful while the parent stays there; a parent free to move is
    dragged to wherever its own constraints put it and leaves its children behind, which is the
    clumping the narrowing avoids.
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
                    position_bbox = get_on_parent_position_bbox(obj, parent_world_bbox, asset_to_bbox)
                    # Narrowing keeps a parent close to where it is sampled, so its children are
                    # not left behind when the solve pulls it towards its own constraints.
                    bounds: AxisAlignedBoundingBox | None = _get_initialization_bounds(obj)
                    if bounds is not None:
                        position_bbox = _narrow_to_bounds(position_bbox, bounds)
                    positions[obj] = sample_position_in_bbox(position_bbox, generator)
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


def _bounds_from_position_limits_box(relation: PositionLimitsBox) -> AxisAlignedBoundingBox:
    """Read the region a PositionLimitsBox confines an object's position to.

    Axes the relation leaves unset become infinite, so they survive intersection untouched.
    """
    return AxisAlignedBoundingBox(
        min_point=(
            relation.x_min if relation.x_min is not None else -math.inf,
            relation.y_min if relation.y_min is not None else -math.inf,
            relation.z_min if relation.z_min is not None else -math.inf,
        ),
        max_point=(
            relation.x_max if relation.x_max is not None else math.inf,
            relation.y_max if relation.y_max is not None else math.inf,
            relation.z_max if relation.z_max is not None else math.inf,
        ),
    )


# Relation types whose constraint can be expressed as a box of allowed positions, and so can narrow
# the region an object is seeded in. Anything absent from this table is simply not used for
# narrowing; it still shapes the layout through its loss during the solve.
_BOUNDS_FACTORY_BY_RELATION_TYPE: dict[type[RelationBase], Callable[[RelationBase], AxisAlignedBoundingBox]] = {
    PositionLimitsBox: _bounds_from_position_limits_box,
}


def _get_initialization_bounds(obj: PlaceableAsset) -> AxisAlignedBoundingBox | None:
    """Return the box obj should be seeded within, or None when nothing narrows it.

    Every supported relation on obj contributes a box of allowed positions, and the result is
    their intersection.
    """
    bounds: AxisAlignedBoundingBox | None = None
    for relation in obj.get_relations():
        for relation_type, bounds_factory in _BOUNDS_FACTORY_BY_RELATION_TYPE.items():
            if isinstance(relation, relation_type):
                relation_bounds = bounds_factory(relation)
                bounds = relation_bounds if bounds is None else bounds.intersected(relation_bounds)
                break
    return bounds


def _narrow_to_bounds(position_bbox: AxisAlignedBoundingBox, bounds: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
    """Return the part of position_bbox that bounds allows.

    Axes where the two are disjoint have no position satisfying both, and collapse to the point of
    position_bbox nearest bounds, which is the shortest reconciliation available.
    """
    intersection = position_bbox.intersected(bounds)
    nearest_allowed = torch.clamp(intersection.center, min=position_bbox.min_point, max=position_bbox.max_point)
    is_empty = intersection.min_point > intersection.max_point
    return AxisAlignedBoundingBox(
        min_point=torch.where(is_empty, nearest_allowed, intersection.min_point),
        max_point=torch.where(is_empty, nearest_allowed, intersection.max_point),
    )
