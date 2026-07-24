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

    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class PlaceableAsset(Asset, ABC):
    """Asset whose root pose can be constrained by spatial relations."""

    def __init__(self, name: str, tags: list[str] | None = None, **kwargs) -> None:
        super().__init__(name=name, tags=tags, **kwargs)
        self.initial_pose: Pose | PoseRange | PosePerEnv | None = None
        self._pose_event_cfg: EventTermCfg | None = None
        """Reset event restoring this asset's root pose; ``None`` until a pose with a reset event is set."""
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

    def set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv, create_reset_event: bool = True) -> None:
        """Set the configured root pose and, unless disabled, (re)build its reset event.

        Accepts a fixed ``Pose``, a ``PoseRange`` (randomized on reset), or a ``PosePerEnv``
        (a distinct pose per environment); the interpretation is subclass-specific.

        Args:
            pose: The root pose(s) to store.
            create_reset_event: When True (default), rebuild the reset event that restores this
                pose on every environment reset. Pass False to set only the scene-construction
                pose (see ``set_spawn_pose``).
        """
        self._materialize_pose_state(pose)
        if create_reset_event:
            self._pose_event_cfg = self._build_reset_event()

    def set_spawn_pose(self, pose: Pose) -> None:
        """Set the scene-construction pose without registering a reset event."""
        self.set_initial_pose(pose, create_reset_event=False)

    def _materialize_pose_state(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Store the configured pose; subclasses also materialize any derived construction config."""
        self.initial_pose = pose

    def _build_reset_event(self) -> EventTermCfg | None:
        """Build the reset event that restores this asset's pose (and velocity, where applicable).

        The base asset owns no reset event; subclasses that reset their root on every environment
        reset override this to return the concrete ``EventTermCfg``.
        """

    @staticmethod
    def _collapse_pose_to_single(pose: Pose | PoseRange | PosePerEnv | None) -> Pose | None:
        """Collapse a configured pose to a single ``Pose`` for construction and bounds.

        ``PosePerEnv`` collapses to env 0, ``PoseRange`` to its midpoint, and ``None`` stays ``None``.
        """
        if pose is None:
            return None
        if isinstance(pose, PosePerEnv):
            return pose.poses[0]
        if isinstance(pose, PoseRange):
            return pose.get_midpoint()
        return pose

    def _get_initial_pose_as_pose(self) -> Pose | None:
        """Return the resolved root pose (relation- or reference-derived) as a single ``Pose``."""
        return self._collapse_pose_to_single(self.get_initial_pose())

    def layout_pose_to_scene_writes(self, layout_pose: Pose) -> list[tuple[str, Pose]]:
        """Return the ``(scene entity name, env-local pose)`` writes that realize a solved layout pose.

        A simple asset places only its own root; a compound asset (e.g. a robot on a separate
        stand) overrides this to also place auxiliary prims that move with the root.
        """
        return [(self.get_scene_key(), layout_pose)]

    def has_pose_reset_event(self) -> bool:
        """Return whether the asset owns a root-pose reset event."""
        return self._pose_event_cfg is not None

    def has_unplaced_auxiliary_prims(self) -> bool:
        """Whether this asset owns auxiliary scene prims that per-environment reset does not reposition.

        Defaults to False. A compound asset whose ``layout_pose_to_scene_writes`` does not yet emit
        writes for all of its prims (e.g. Droid's static stand) overrides this to True so relation
        placement can reject it loudly instead of silently orphaning those prims.
        """
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
        """Return this asset's collision mesh, or ``None`` to fall back to the axis-aligned bounds.

        Concrete (not abstract) so assets without a mesh simply keep the ``None`` default.
        """
