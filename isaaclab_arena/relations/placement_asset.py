# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared model for assets whose poses are relation-solved."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.relations import IsAnchor, Relation, RelationBase, UnaryRelation
from isaaclab_arena.utils.bounding_box import quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange

if TYPE_CHECKING:
    import trimesh

    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class PlacementAsset(Asset, ABC):
    """Asset whose root pose can be constrained by spatial relations."""

    def __init__(self, name: str, tags: list[str] | None = None, **kwargs) -> None:
        super().__init__(name=name, tags=tags, **kwargs)
        self.initial_pose: Pose | PoseRange | PosePerEnv | None = None
        self.relations: list[RelationBase] = []
        # None delegates collision-mode selection to the solver.
        self.collision_mode: CollisionMode | None = None
        # Whether to replace a non-watertight collision mesh with its convex hull.
        self.repair_collision_mesh_non_watertight = True

    def add_relation(self, relation: RelationBase) -> None:
        """Attach a relation to the asset."""
        self.relations.append(relation)

    def get_relations(self) -> list[RelationBase]:
        """Return all relations attached to the asset."""
        return self.relations

    def get_spatial_relations(self) -> list[RelationBase]:
        """Return spatial constraints, excluding placement markers."""
        return [relation for relation in self.relations if isinstance(relation, (Relation, UnaryRelation))]

    @property
    def is_anchor(self) -> bool:
        """Return whether the asset is fixed during relation solving."""
        return any(isinstance(relation, IsAnchor) for relation in self.relations)

    def get_initial_pose(self) -> Pose | PoseRange | PosePerEnv | None:
        """Return the configured root pose."""
        return self.initial_pose

    def set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Set the configured root pose.

        Accepts a fixed ``Pose``, a ``PoseRange`` (randomized on reset), or a ``PosePerEnv``
        (a distinct pose per environment); the interpretation is subclass-specific.
        """
        self.initial_pose = pose

    def set_spawn_pose(self, pose: Pose) -> None:
        """Set the root pose used when constructing the scene."""
        self.set_initial_pose(pose)

    def layout_pose_to_scene_writes(self, layout_pose: Pose) -> list[tuple[str, Pose]]:
        """Return the ``(scene entity name, env-local pose)`` writes that realize a solved layout pose.

        A simple asset places only its own root; a compound asset (e.g. a robot on a separate
        stand) overrides this to also place auxiliary prims that move with the root.
        """
        return [(self.get_scene_name(), layout_pose)]

    def has_pose_reset_event(self) -> bool:
        """Return whether the asset owns a root-pose reset event."""
        return False

    @abstractmethod
    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return root-relative axis-aligned bounds."""

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return bounds transformed by a fixed root pose with a quarter-turn Z rotation.

        Unset, ranged, and per-environment poses leave the root-relative bounds unchanged.
        """
        bounding_box = self.get_bounding_box()
        initial_pose = self.get_initial_pose()
        if not isinstance(initial_pose, Pose):
            return bounding_box
        quarters = quaternion_to_90_deg_z_quarters(initial_pose.rotation_xyzw)
        return bounding_box.rotated_90_around_z(quarters).translated(initial_pose.position_xyz)

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return this asset's collision mesh, or ``None`` if it has none."""
