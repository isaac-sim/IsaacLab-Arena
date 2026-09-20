# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import PositionLimitsBox, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


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


def get_initialization_bounds(obj: PlaceableAsset) -> AxisAlignedBoundingBox | None:
    """Return the box obj should be seeded within, or None when nothing narrows it.

    Every supported relation on obj contributes a box of allowed positions, and the result is
    their intersection. Seeding an object inside its parent's footprint is only useful while the
    parent stays where it was sampled; a parent free to move is dragged to wherever its own
    constraints put it and leaves its children behind, which is the clumping this avoids.
    """
    bounds: AxisAlignedBoundingBox | None = None
    for relation in obj.get_relations():
        for relation_type, bounds_factory in _BOUNDS_FACTORY_BY_RELATION_TYPE.items():
            if isinstance(relation, relation_type):
                relation_bounds = bounds_factory(relation)
                bounds = relation_bounds if bounds is None else bounds.intersected(relation_bounds)
                break
    return bounds
