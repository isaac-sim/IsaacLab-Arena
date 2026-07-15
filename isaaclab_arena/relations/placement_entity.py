# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared model for entities whose poses are relation-solved."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.relations import IsAnchor, Relation, RelationBase, UnaryRelation
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange

if TYPE_CHECKING:
    import trimesh

    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class PlacementEntity(Asset, ABC):
    """Asset whose root pose can be constrained by spatial relations."""

    def __init__(self, name: str, tags: list[str] | None = None, **kwargs) -> None:
        super().__init__(name=name, tags=tags, **kwargs)
        self.initial_pose: Pose | PoseRange | PosePerEnv | None = None
        self.relations: list[RelationBase] = []
        # None delegates collision-mode selection to the solver.
        self.collision_mode: CollisionMode | None = None
        # Mesh collision uses a convex hull when source geometry is not watertight.
        self.repair_collision_mesh_non_watertight = True

    def add_relation(self, relation: RelationBase) -> None:
        """Attach a relation to the entity."""
        self.relations.append(relation)

    def get_relations(self) -> list[RelationBase]:
        """Return all relations attached to the entity."""
        return self.relations

    def get_spatial_relations(self) -> list[RelationBase]:
        """Return spatial constraints, excluding placement markers."""
        return [relation for relation in self.relations if isinstance(relation, (Relation, UnaryRelation))]

    @property
    def is_anchor(self) -> bool:
        """Return whether the entity is fixed during relation solving."""
        return any(isinstance(relation, IsAnchor) for relation in self.relations)

    def get_initial_pose(self) -> Pose | PoseRange | PosePerEnv | None:
        """Return the configured root pose."""
        return self.initial_pose

    def set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Set the configured root pose."""
        self.initial_pose = pose

    def set_placement_initial_pose(self, pose: Pose) -> None:
        """Set a solved root pose without changing reset ownership."""
        self.set_initial_pose(pose)

    def has_pose_reset_event(self) -> bool:
        """Return whether the entity owns a root-pose reset event."""
        return False

    def supports_per_env_initial_pose(self) -> bool:
        """Return whether one configured pose may be stored per environment."""
        return True

    @abstractmethod
    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return root-relative axis-aligned bounds."""

    @abstractmethod
    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return axis-aligned bounds transformed by the root pose."""

    @abstractmethod
    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return a root-relative collision mesh when available."""
