# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import torch

from isaaclab_arena.relations.relations import Relation, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
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

    def get_spatial_relations(self) -> list[Relation]:
        """Get only spatial relations (On, NextTo, etc.), excluding markers like IsAnchor."""
        return [r for r in self.relations if isinstance(r, Relation)]

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        return self.bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get bounding box in world coordinates (local bbox + position offset)."""
        pos = self.initial_pose.position_xyz if self.initial_pose else (0, 0, 0)
        return AxisAlignedBoundingBox(
            min_point=(
                self.bounding_box.min_point[0] + pos[0],
                self.bounding_box.min_point[1] + pos[1],
                self.bounding_box.min_point[2] + pos[2],
            ),
            max_point=(
                self.bounding_box.max_point[0] + pos[0],
                self.bounding_box.max_point[1] + pos[1],
                self.bounding_box.max_point[2] + pos[2],
            ),
        )

    def get_corners_aabb(self, pos: torch.Tensor) -> torch.Tensor:
        return self.bounding_box.get_corners_at(pos)

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_initial_pose(self) -> Pose | None:
        return self.initial_pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None
