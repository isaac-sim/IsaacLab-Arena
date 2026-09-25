# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import On, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class InitializerType(Enum):
    """Available placement initialization strategies."""

    ANCHOR = "anchor"
    """Seed against the footprint of the first anchor at or above each object's On parent."""


class InitializerBase(ABC):
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


def get_world_bbox_at_initial_pose(
    obj: PlaceableAsset, asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox]
) -> AxisAlignedBoundingBox:
    """Return obj's local bbox translated to its fixed initial pose."""
    initial_pose = obj.get_initial_pose()
    assert isinstance(
        initial_pose, Pose
    ), f"Object '{obj.name}' must have a fixed Pose to use its env bbox, got {type(initial_pose).__name__}."
    return asset_to_bbox[obj].translated(initial_pose.position_xyz)


def sample_uniform_or_midpoint(low: float, high: float, generator: torch.Generator | None = None) -> float:
    """Sample uniformly from [low, high], falling back to the midpoint when the interval is empty.

    An empty interval means the caller's constraints cannot all hold, which happens for example
    when a child is wider than the surface it sits on. The midpoint keeps the seed centred on the
    intended region and lets the solver resolve the conflict.
    """
    if low >= high:
        return float((low + high) / 2.0)
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


class AnchorInitializer(InitializerBase):
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
                positions[obj] = _fixed_anchor_position(obj)
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


def _fixed_anchor_position(obj: PlaceableAsset) -> tuple[float, float, float]:
    """Return an anchor's fixed spawn position."""
    initial_pose = obj.get_initial_pose()
    assert isinstance(
        initial_pose, Pose
    ), f"Anchor object '{obj.name}' must have a fixed Pose before placement, got {type(initial_pose).__name__}."
    return initial_pose.position_xyz


def sample_on_parent(
    obj: PlaceableAsset,
    parent_world_bbox: AxisAlignedBoundingBox,
    asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox],
    generator: torch.Generator | None = None,
) -> tuple[float, float, float]:
    """Sample a position for obj on top of parent_world_bbox.

    X and Y are drawn from the parent's footprint inset by the child's extents, and Z is set so
    the child's bottom face rests on the parent's top surface plus the relation's clearance.

    Args:
        obj: The object being seeded; must carry an ``On`` relation.
        parent_world_bbox: World-space bbox of the parent, shape (1, 3).
        asset_to_bbox: Local bounding box per object for the env this candidate belongs to.
        generator: Optional RNG generator for reproducible sampling.
    """
    on_relation = get_relation(obj, On)
    child_bbox = asset_to_bbox[obj]

    child_min, child_max = child_bbox.min_point[0], child_bbox.max_point[0]
    if on_relation.overlap:
        # Intersection compares the child's far edge with the parent's near edge.
        child_min, child_max = child_max, child_min

    position_xy: list[float] = []
    for axis in (0, 1):
        low = float(parent_world_bbox.min_point[0, axis]) - float(child_min[axis])
        high = float(parent_world_bbox.max_point[0, axis]) - float(child_max[axis])
        if low >= high:
            # Child does not fit on the parent along this axis; seed at the parent's center.
            position_xy.append(float(parent_world_bbox.center[0, axis]))
            continue
        position_xy.append(sample_uniform_or_midpoint(low, high, generator))

    # Convert from child-origin Z to child-bottom Z so the bottom face lands on the parent top.
    z = float(parent_world_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2])
    return (position_xy[0], position_xy[1], z)


_INITIALIZERS_BY_TYPE: dict[InitializerType, type[InitializerBase]] = {
    InitializerType.ANCHOR: AnchorInitializer,
}


def create_initializer(initializer_type: InitializerType) -> InitializerBase:
    """Return a new initializer of the requested type."""
    assert initializer_type in _INITIALIZERS_BY_TYPE, f"No initializer registered for {initializer_type}."
    return _INITIALIZERS_BY_TYPE[initializer_type]()
