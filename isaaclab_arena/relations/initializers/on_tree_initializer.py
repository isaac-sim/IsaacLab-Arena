# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.initializers.placement_initializer_base import PlacementInitializerBase
from isaaclab_arena.relations.initializers.sampling import (
    get_child_bbox_given_parent_position,
    get_initial_pose_or_assert_fail,
    get_world_bbox_at_initial_pose,
    sample_position_in_bbox,
)
from isaaclab_arena.relations.relations import On, PositionLimitsBox, RelationBase, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class OnTreeInitializer(PlacementInitializerBase):
    """Generates initial positions for objects by walking down the ``On`` tree in the relation graph.

    Each object is sampled inside the footprint its own parent was just sampled at.
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

        # The position sampled for each object so far, in the world frame.
        positions: dict[PlaceableAsset, tuple[float, float, float]] = {}
        # Bounding boxes of the objects sampled so far, in the world frame, i.e. their local
        # bounding boxes translated by the position each one was just sampled at.
        sampled_world_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox] = {}
        ordered_objects: list[PlaceableAsset] = _order_parents_before_children(objects, anchor_objects)
        for obj in ordered_objects:
            if obj in anchor_objects:
                positions[obj] = get_initial_pose_or_assert_fail(obj).position_xyz
            else:
                on_relation: On | None = get_relation(obj, On)
                if on_relation is None:
                    positions[obj] = fallback_position
                else:
                    parent_world_bbox: AxisAlignedBoundingBox = sampled_world_bboxes[on_relation.parent]
                    # Take the positions that keep obj on its parent, shrink them to what obj's
                    # other relations allow, and draw one. Narrowing keeps a parent close to where
                    # it is sampled, so its children are not left behind when the solve pulls it
                    # towards its own constraints.
                    child_bbox_given_parent_position = get_child_bbox_given_parent_position(
                        obj, parent_world_bbox, asset_to_bbox
                    )
                    other_bounds = _get_bounds_from_other_supported_relations(obj)
                    sampling_bbox = _maybe_narrow_bounds(child_bbox_given_parent_position, other_bounds)
                    positions[obj] = sample_position_in_bbox(sampling_bbox, generator)
            sampled_world_bboxes[obj] = asset_to_bbox[obj].translated(positions[obj])
        # Return the positions in the order the caller supplied the objects, rather than in the
        # parents-first order this method sampled them in.
        return {obj: positions[obj] for obj in objects}


def _order_parents_before_children(
    objects: list[PlaceableAsset],
    anchor_objects: set[PlaceableAsset],
) -> list[PlaceableAsset]:
    """Return objects ordered so every ``On`` parent precedes its children."""
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
# TODO(alexmillane, 2026.09.27): Expand this list of relations used for narrowing as required.
_BOUNDS_FACTORY_BY_RELATION_TYPE: dict[type[RelationBase], Callable[[RelationBase], AxisAlignedBoundingBox]] = {
    PositionLimitsBox: _bounds_from_position_limits_box,
}


def _get_bounds_from_other_supported_relations(obj: PlaceableAsset) -> AxisAlignedBoundingBox:
    """Return the box obj's supported non-``On`` relations confine its position to.

    Every supported relation contributes a box of allowed positions and the result is their
    intersection. An object with no such relation gets an unbounded box, which narrows nothing.
    """
    bounds = AxisAlignedBoundingBox(
        min_point=(-math.inf, -math.inf, -math.inf), max_point=(math.inf, math.inf, math.inf)
    )
    for relation in obj.get_relations():
        for relation_type, bounds_factory in _BOUNDS_FACTORY_BY_RELATION_TYPE.items():
            if isinstance(relation, relation_type):
                bounds = bounds.intersected(bounds_factory(relation))
                break
    return bounds


def _maybe_narrow_bounds(
    child_bbox_given_parent_position: AxisAlignedBoundingBox, other_bounds: AxisAlignedBoundingBox
) -> AxisAlignedBoundingBox:
    """Return the two boxes intersected, or the first one alone when they do not overlap.

    An empty intersection means no position sits on the parent and satisfies the other relations,
    so the other relations are dropped and the object is seeded on its parent as if they were
    absent. The solve reconciles them from there.
    """
    narrowed = child_bbox_given_parent_position.intersected(other_bounds)
    if bool((narrowed.min_point > narrowed.max_point).any()):
        return child_bbox_given_parent_position
    return narrowed
