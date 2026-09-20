# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strategies for seeding the relation solver's optimization variables."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import On, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

Position = tuple[float, float, float]
EnvBoundingBoxes = dict["PlaceableAsset", AxisAlignedBoundingBox]


class InitializerBase(ABC):
    """Produces the starting positions the relation solver optimizes from.

    The solver minimizes a hinge loss that is exactly zero once a relation is satisfied, so
    it stops at the first feasible point it reaches. Where a candidate starts therefore decides
    which part of the feasible set it lands in, and an initializer's job is to spread candidates
    over that set rather than to find a solution itself.
    """

    @abstractmethod
    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: EnvBoundingBoxes,
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, Position]:
        """Return one starting position per object for a single solver candidate.

        Args:
            objects: Every object taking part in the solve, anchors included.
            anchor_objects: The subset of objects that stay at their fixed initial pose.
            env_bboxes: Per-object local bounding boxes for this env, each of shape (1, 3).
            generator: RNG for reproducible sampling. None uses PyTorch's global RNG.

        Returns:
            Starting position for every object in ``objects``.
        """


def get_world_bbox_at_initial_pose(obj: PlaceableAsset, env_bboxes: EnvBoundingBoxes) -> AxisAlignedBoundingBox:
    """Return obj's local bbox translated to its fixed initial pose."""
    initial_pose = obj.get_initial_pose()
    assert isinstance(
        initial_pose, Pose
    ), f"Object '{obj.name}' must have a fixed Pose to use its env bbox, got {type(initial_pose).__name__}."
    return env_bboxes[obj].translated(initial_pose.position_xyz)


def sample_uniform(low: float, high: float, generator: torch.Generator | None = None) -> float:
    """Sample uniformly from [low, high], returning the midpoint when the interval is empty."""
    if low >= high:
        return float((low + high) / 2.0)
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


class AnchorInitializer(InitializerBase):
    """Seeds every object against an anchor's footprint.

    Objects with an ``On`` relation are sampled inside the footprint of the anchor at or above
    their parent; all others start at the first anchor's center. Only one level of ``On``
    indirection is resolved, so a child of a non-anchor parent is seeded across the whole anchor
    rather than across its actual parent.
    """

    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: EnvBoundingBoxes,
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, Position]:
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        anchor_bbox = get_world_bbox_at_initial_pose(first_anchor, env_bboxes)
        center = anchor_bbox.center[0]
        anchor_center = (float(center[0]), float(center[1]), float(center[2]))

        positions: dict[PlaceableAsset, Position] = {}
        for obj in objects:
            if obj in anchor_objects:
                positions[obj] = _fixed_anchor_position(obj)
            elif get_relation(obj, On) is not None:
                parent_bbox = self._get_on_parent_world_bbox(obj, anchor_objects, anchor_bbox, env_bboxes)
                positions[obj] = sample_on_parent(obj, parent_bbox, env_bboxes, generator)
            else:
                positions[obj] = anchor_center
        return positions

    @staticmethod
    def _get_on_parent_world_bbox(
        obj: PlaceableAsset,
        anchor_objects: set[PlaceableAsset],
        anchor_bbox: AxisAlignedBoundingBox,
        env_bboxes: EnvBoundingBoxes,
    ) -> AxisAlignedBoundingBox:
        """Resolve the world bbox of an On relation's parent for initialization purposes.

        If the parent is an anchor, return its world bbox directly. If the parent is a non-anchor
        with its own On(anchor) relation, use the anchor's world bbox as a proxy. Only one level of
        indirection is resolved; deeper chains fall back to anchor_bbox.
        """
        parent = get_relation(obj, On).parent
        if parent in anchor_objects:
            return get_world_bbox_at_initial_pose(parent, env_bboxes)
        for relation in parent.get_relations():
            if isinstance(relation, On) and relation.parent in anchor_objects:
                return get_world_bbox_at_initial_pose(relation.parent, env_bboxes)
        return anchor_bbox


def _fixed_anchor_position(obj: PlaceableAsset) -> Position:
    """Return an anchor's fixed spawn position."""
    initial_pose = obj.get_initial_pose()
    assert isinstance(
        initial_pose, Pose
    ), f"Anchor object '{obj.name}' must have a fixed Pose before placement, got {type(initial_pose).__name__}."
    return initial_pose.position_xyz


def sample_on_parent(
    obj: PlaceableAsset,
    parent_world_bbox: AxisAlignedBoundingBox,
    env_bboxes: EnvBoundingBoxes,
    generator: torch.Generator | None = None,
) -> Position:
    """Sample a position for obj on top of parent_world_bbox.

    X and Y are drawn from the parent's footprint inset by the child's extents, and Z is set so
    the child's bottom face rests on the parent's top surface plus the relation's clearance.

    Args:
        obj: The object being seeded; must carry an ``On`` relation.
        parent_world_bbox: World-space bbox of the parent, shape (1, 3).
        env_bboxes: Per-object local bounding boxes for this env.
        generator: Optional RNG generator for reproducible sampling.
    """
    on_relation = get_relation(obj, On)
    child_bbox = env_bboxes[obj]

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
        position_xy.append(sample_uniform(low, high, generator))

    # Convert from child-origin Z to child-bottom Z so the bottom face lands on the parent top.
    z = float(parent_world_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2])
    return (position_xy[0], position_xy[1], z)
