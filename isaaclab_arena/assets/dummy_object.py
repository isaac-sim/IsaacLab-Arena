# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import torch

from isaaclab_arena.utils.bounding_box import BoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.relations import Relation


class DummyObject:
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(
        self,
        name: str,
        bounding_box: BoundingBox,
        initial_pose: Pose | None = None,
        relations: list[Relation] = [],
        is_fixed: bool = False,
        **kwargs,
    ):
        self.name = name
        self.initial_pose = initial_pose
        self.bounding_box = bounding_box
        assert self.bounding_box is not None
        self.relations = []
        self.is_fixed = is_fixed

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def get_relations(self) -> list[Relation]:
        return self.relations

    def get_bounding_box(self) -> BoundingBox:
        return self.bounding_box

    def get_corners_aabb_axis_aligned(self, pos: torch.Tensor) -> torch.Tensor:
        return self.bounding_box.get_corners_aabb_axis_aligned(pos)

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_initial_pose(self) -> Pose | None:
        return self.initial_pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None
