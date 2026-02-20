# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol, runtime_checkable

from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PoseRange


@runtime_checkable
class PlaceableEntity(Protocol):
    """Structural interface for any object the relation solver can position.

    Any class that implements these methods (e.g. Object, ObjectReference,
    EmbodimentBase, DummyObject) is automatically a PlaceableEntity.
    """

    @property
    def name(self) -> str | None:
        """Name of the placeable object."""
        ...

    def add_relation(self, relation: RelationBase) -> None:
        """Add a spatial relation or marker to this object."""
        ...

    def get_relations(self) -> list[RelationBase]:
        """Get all relations (spatial relations + markers like IsAnchor)."""
        ...

    def get_spatial_relations(self) -> list[RelationBase]:
        """Get only spatial relations (On, NextTo, AtPosition, etc.), excluding markers."""
        ...

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        ...

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get bounding box in world coordinates."""
        ...

    def set_initial_pose(self, pose: Pose | PoseRange) -> None:
        """Set the initial pose of this object."""
        ...

    def get_initial_pose(self) -> Pose | PoseRange | None:
        """Get the initial pose of this object."""
        ...
