# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.initializers.placement_initializer_base import PlacementInitializerBase
from isaaclab_arena.relations.initializers.relation_sampling_strategies import (
    _bounding_box_from_partial_limits,
    get_sampling_strategy,
)
from isaaclab_arena.relations.initializers.sampling import (
    get_child_bbox_given_parent_position,
    get_initial_pose_or_assert_fail,
    get_world_bbox_at_initial_pose,
    sample_position_in_bbox,
)
from isaaclab_arena.relations.relations import On, get_relation
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
        # Sampling a child needs its parent's position, so visit parents first.
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
                    # The positions that keep obj on its parent.
                    child_bbox_given_parent_position = get_child_bbox_given_parent_position(
                        obj, parent_world_bbox, asset_to_bbox
                    )
                    # The positions obj's other relations allow.
                    other_bounds = _get_bounds_from_other_supported_relations(
                        obj, asset_to_bbox[obj], sampled_world_bboxes
                    )
                    # Both at once, which keeps obj near where the solve will pull it.
                    sampling_bbox = _maybe_narrow_bounds(child_bbox_given_parent_position, other_bounds)
                    # One position drawn from what is left.
                    positions[obj] = sample_position_in_bbox(sampling_bbox, generator)
            sampled_world_bboxes[obj] = asset_to_bbox[obj].translated(positions[obj])
        # Return the positions in the order the caller supplied the objects, rather than in the
        # parents-first order this method sampled them in.
        return {obj: positions[obj] for obj in objects}


def _order_parents_before_children(
    objects: list[PlaceableAsset],
    anchor_objects: set[PlaceableAsset],
) -> list[PlaceableAsset]:
    """Order support parents first, preferring available sampling references across tree levels."""
    ordered: list[PlaceableAsset] = []
    remaining: list[PlaceableAsset] = []
    dependencies: dict[PlaceableAsset, list[PlaceableAsset]] = {}
    on_children: dict[PlaceableAsset, list[PlaceableAsset]] = {}
    for obj in objects:
        on_relation = get_relation(obj, On)
        if obj in anchor_objects or on_relation is None:
            ordered.append(obj)
        else:
            remaining.append(obj)
            on_children.setdefault(on_relation.parent, []).append(obj)
            dependencies[obj] = []
            for relation in obj.get_relations():
                strategy = get_sampling_strategy(relation)
                if strategy is not None:
                    dependencies[obj].extend(strategy.sampling_dependencies(relation))

    ordered_set = set(ordered)
    while remaining:
        # Recompute the frontier after each object: a newly eligible descendant may supply a
        # reference needed by an object at a shallower level.
        ready = [obj for obj in remaining if get_relation(obj, On).parent in ordered_set]
        assert ready, (
            "On relations must form a forest rooted at anchors, but no parent could be resolved for "
            f"{[obj.name for obj in remaining]}. Check for a cycle or a parent outside the placement set."
        )
        missing = {}
        for obj in remaining:
            missing[obj] = [reference for reference in dependencies[obj] if reference not in ordered_set]
        references_ready = [obj for obj in ready if not missing[obj]]
        if references_ready:
            selected = references_ready[0]
        else:
            # Minimize skipped constraints, then prefer opening a support subtree containing
            # more waiting references. This is a heuristic, not an optimal graph ordering.
            # min retains caller order for equal scores, making cycle breaking reproducible.
            selected = min(
                ready,
                key=lambda obj: (len(missing[obj]), -_count_waiting_dependencies(obj, on_children, missing)),
            )
        ordered.append(selected)
        ordered_set.add(selected)
        remaining.remove(selected)
    return ordered


def _count_waiting_dependencies(
    obj: PlaceableAsset,
    on_children: dict[PlaceableAsset, list[PlaceableAsset]],
    missing: dict[PlaceableAsset, list[PlaceableAsset]],
) -> int:
    """Count waiting relation references to an object or its support descendants."""
    subtree = set()
    pending = [obj]
    while pending:
        descendant = pending.pop()
        if descendant not in subtree:
            subtree.add(descendant)
            pending.extend(on_children.get(descendant, ()))
    count = 0
    for subject, references in missing.items():
        if subject is not obj:
            count += sum(reference in subtree for reference in references)
    return count


def _get_bounds_from_other_supported_relations(
    obj: PlaceableAsset,
    child_bbox: AxisAlignedBoundingBox,
    sampled_world_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
) -> AxisAlignedBoundingBox:
    """Return the box obj's supported non-``On`` relations confine its position to.

    Every supported relation contributes a box of allowed positions and the result is their
    intersection. Relations with unsampled dependencies are skipped to break cycles.

    Args:
        obj: Object whose starting position is being sampled.
        child_bbox: Subject bounds relative to its origin, expressed along world axes.
        sampled_world_bboxes: World bounds for objects already sampled.

    Returns:
        Intersected bounds, or an unbounded box when no supported relation is available.
    """
    bounds = _bounding_box_from_partial_limits((None, None, None), (None, None, None))
    for relation in obj.get_relations():
        strategy = get_sampling_strategy(relation)
        if strategy is None:
            continue
        if all(reference in sampled_world_bboxes for reference in strategy.sampling_dependencies(relation)):
            bounds = bounds.intersected(strategy.sampling_bounds(relation, child_bbox, sampled_world_bboxes))
    return bounds


def _maybe_narrow_bounds(
    child_bbox_given_parent_position: AxisAlignedBoundingBox, other_bounds: AxisAlignedBoundingBox
) -> AxisAlignedBoundingBox:
    """Return the two boxes intersected, falling back per axis where they do not overlap.

    An axis with an empty intersection has no position that both sits on the parent and satisfies
    the other relations, so that axis keeps the on-parent range as if those relations were absent.
    The solve reconciles it from there. ``On`` pins Z to a single value, so a relation that also
    bounds Z would otherwise discard the X and Y narrowing along with it.
    """
    narrowed = child_bbox_given_parent_position.intersected(other_bounds)
    empty_axes = narrowed.min_point > narrowed.max_point
    return AxisAlignedBoundingBox(
        min_point=torch.where(empty_axes, child_bbox_given_parent_position.min_point, narrowed.min_point),
        max_point=torch.where(empty_axes, child_bbox_given_parent_position.max_point, narrowed.max_point),
    )
