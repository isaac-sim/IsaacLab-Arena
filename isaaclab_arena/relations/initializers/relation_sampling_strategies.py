# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import AlignWith, PositionLimitsBox, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class RelationSamplingStrategy(ABC):
    """Supply dependencies and allowed origin bounds for initializing a relation."""

    def sampling_dependencies(self, relation: RelationBase) -> tuple[PlaceableAsset, ...]:
        """Return references whose world bounds must be available before sampling_bounds."""
        return ()

    @abstractmethod
    def sampling_bounds(
        self,
        relation: RelationBase,
        subject_bbox: AxisAlignedBoundingBox,
        sampled_world_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    ) -> AxisAlignedBoundingBox:
        """Return allowed subject-origin bounds once all sampling dependencies are available.

        Args:
            relation: Relation being initialized.
            subject_bbox: Oriented subject bounds relative to its origin, along world axes.
            sampled_world_bboxes: World bounds of objects already sampled.
        """


class PositionLimitsBoxSamplingStrategy(RelationSamplingStrategy):
    """Narrow origins to absolute position limits without reference dependencies."""

    def sampling_bounds(self, relation, subject_bbox, sampled_world_bboxes):
        return _bounds_from_position_limits_box(relation)


class AlignWithSamplingStrategy(RelationSamplingStrategy):
    """Narrow origins to an alignment band after its reference has been sampled."""

    def sampling_dependencies(self, relation):
        return (relation.parent,)

    def sampling_bounds(self, relation, subject_bbox, sampled_world_bboxes):
        return _bounds_from_align_with(relation, subject_bbox, sampled_world_bboxes[relation.parent])


SAMPLING_STRATEGIES: dict[type[RelationBase], RelationSamplingStrategy] = {
    PositionLimitsBox: PositionLimitsBoxSamplingStrategy(),
    AlignWith: AlignWithSamplingStrategy(),
}


def get_sampling_strategy(relation: RelationBase) -> RelationSamplingStrategy | None:
    """Find the registered strategy for the relation's most specific supported base class."""
    for relation_type in type(relation).__mro__:
        if relation_type in SAMPLING_STRATEGIES:
            return SAMPLING_STRATEGIES[relation_type]
    return None


# NOTE(alexmillane, 2026.10.02): Partial limits are not currently supported by our bounding box
# class, and would break some of the methods there. I'm adding local support to this file, where we
# need it for intersection with narrowing bounds, however, if we end up needing this in multiple
# places consider expanding our bounding box class to handle it (or add another class).
def _bounding_box_from_partial_limits(
    min_point: tuple[float | None, float | None, float | None],
    max_point: tuple[float | None, float | None, float | None],
) -> AxisAlignedBoundingBox:
    """Build a box from per-axis limits where None means unbounded on that side.

    Unset sides become infinite, which is the identity for ``intersected``, so they narrow
    nothing. The result may therefore be unbounded: that is fine for ``intersected`` and
    ``overlaps``, which only compare, but not for ``size``, ``center`` or the rotation helpers.
    """
    return AxisAlignedBoundingBox(
        min_point=tuple(-math.inf if limit is None else limit for limit in min_point),
        max_point=tuple(math.inf if limit is None else limit for limit in max_point),
    )


def _bounds_from_position_limits_box(relation: PositionLimitsBox) -> AxisAlignedBoundingBox:
    """Read the region a PositionLimitsBox confines an object's position to."""
    return _bounding_box_from_partial_limits(
        (relation.x_min, relation.y_min, relation.z_min),
        (relation.x_max, relation.y_max, relation.z_max),
    )


def _bounds_from_align_with(
    relation: AlignWith,
    child_bbox: AxisAlignedBoundingBox,
    reference_world_bbox: AxisAlignedBoundingBox,
) -> AxisAlignedBoundingBox:
    """Return origin bounds for alignment to a reference's sampled bounding-box centre.

    Args:
        relation: Alignment axis, signed centre offset, and tolerance band.
        child_bbox: Subject bounds relative to its origin, expressed along world axes.
        reference_world_bbox: Reference bounds at its sampled world position.

    Returns:
        Bounds on the subject origin, unbounded on the other axes.
    """
    axis_index = relation.axis.index
    target_origin_W = (
        reference_world_bbox.center[0, axis_index] - child_bbox.center[0, axis_index] + relation.offset_m
    ).item()
    lower = [None, None, None]
    upper = [None, None, None]
    lower[axis_index] = target_origin_W - relation.tolerance_m
    upper[axis_index] = target_origin_W + relation.tolerance_m
    return _bounding_box_from_partial_limits(tuple(lower), tuple(upper))
