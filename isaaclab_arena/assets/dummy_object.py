# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import torch

from isaaclab_arena.relations.relations import Relation, RelationBase, UnaryRelation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose


class DummyObject:
    """
    Dummy object for testing purposes without Isaac Sim dependencies.
    """

    def __init__(
        self,
        name: str,
        bounding_box: AxisAlignedBoundingBox,
        initial_pose: Pose | None = None,
        relations: list[RelationBase] = [],
        **kwargs,
    ):
        self.name = name
        self.initial_pose = initial_pose
        self.bounding_box = bounding_box
        assert self.bounding_box is not None
        self.relations = []

    def add_relation(self, relation: RelationBase) -> None:
        self.relations.append(relation)

    def get_relations(self) -> list[RelationBase]:
        return self.relations

    def get_spatial_relations(self) -> list[RelationBase]:
        """Get only spatial relations (On, NextTo, AtPosition, etc.), excluding markers like IsAnchor."""
        return [r for r in self.relations if isinstance(r, (Relation, UnaryRelation))]

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        return self.bounding_box

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Get per-environment local bounding boxes (expanded from single bbox)."""
        bbox = self.get_bounding_box()
        return AxisAlignedBoundingBox(
            min_point=bbox.min_point.expand(num_envs, 3),
            max_point=bbox.max_point.expand(num_envs, 3),
        )

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get bounding box in world coordinates (local bbox rotated and translated).

        Only 90° rotations around Z axis are supported.
        """
        if self.initial_pose is None:
            return self.bounding_box
        quarters = quaternion_to_90_deg_z_quarters(self.initial_pose.rotation_xyzw)
        return self.bounding_box.rotated_90_around_z(quarters).translated(self.initial_pose.position_xyz)

    def get_corners_aabb(self, pos: torch.Tensor) -> torch.Tensor:
        return self.bounding_box.get_corners_at(pos)

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_initial_pose(self) -> Pose | None:
        return self.initial_pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None
