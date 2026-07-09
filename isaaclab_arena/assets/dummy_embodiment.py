# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import trimesh

from isaaclab_arena.relations.relations import IsAnchor, Relation, RelationBase, UnaryRelation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose


class DummyEmbodiment:
    """Embodiment stand-in for relation-solver tests without Isaac Sim dependencies."""

    def __init__(
        self,
        name: str,
        bounding_box: AxisAlignedBoundingBox,
        initial_pose: Pose | None = None,
        relations: list[RelationBase] | None = None,
        collision_mesh: trimesh.Trimesh | None = None,
        placement_scene_entity_name: str = "robot",
    ):
        self.name = name
        self.initial_pose = initial_pose
        self.bounding_box = bounding_box
        self.relations = list(relations or [])
        self._collision_mesh = collision_mesh
        self.placement_scene_entity_name = placement_scene_entity_name

    @property
    def placement_kind(self) -> str:
        return "embodiment"

    def add_relation(self, relation: RelationBase) -> None:
        assert not isinstance(relation, IsAnchor), "Embodiment cannot be marked as an anchor"
        self.relations.append(relation)

    def get_relations(self) -> list[RelationBase]:
        return self.relations

    def get_spatial_relations(self) -> list[RelationBase]:
        return [relation for relation in self.relations if isinstance(relation, (Relation, UnaryRelation))]

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        return self.bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        if self.initial_pose is None:
            return self.bounding_box
        quarters = quaternion_to_90_deg_z_quarters(self.initial_pose.rotation_xyzw)
        return self.bounding_box.rotated_90_around_z(quarters).translated(self.initial_pose.position_xyz)

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        return self._collision_mesh

    def get_initial_pose(self) -> Pose | None:
        return self.initial_pose

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    @property
    def is_anchor(self) -> bool:
        return False
