# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared placement interface for scene objects and robot embodiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import trimesh

    from isaaclab_arena.relations.relations import RelationBase
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange

PlacementKind = Literal["object", "embodiment"]


@runtime_checkable
class PlacementEntity(Protocol):
    """Entity whose pose can be solved by the relation placement pipeline."""

    name: str
    relations: list[RelationBase]

    @property
    def is_anchor(self) -> bool:
        """True when this entity is a fixed reference for relation solving."""
        ...

    @property
    def placement_kind(self) -> PlacementKind:
        """Whether solved poses apply to a scene object or an embodiment."""
        ...

    @property
    def placement_scene_entity_name(self) -> str:
        """Isaac Lab scene entity name used when writing solved poses to sim."""
        ...

    def add_relation(self, relation: RelationBase) -> None:
        """Attach a spatial relation or marker to this entity."""
        ...

    def get_relations(self) -> list[RelationBase]:
        """Return all relations attached to this entity."""
        ...

    def get_spatial_relations(self) -> list[RelationBase]:
        """Return spatial constraints, excluding placement markers."""
        ...

    def get_initial_pose(self) -> Pose | PoseRange | PosePerEnv | None:
        """Return the entity's current initial pose, if any."""
        ...

    def set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Set the entity's initial pose from a solved layout."""
        ...

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the local placement footprint relative to the entity origin."""
        ...

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the placement footprint in world coordinates."""
        ...

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return a collision mesh for mesh overlap checks, or None for AABB-only."""
        ...
