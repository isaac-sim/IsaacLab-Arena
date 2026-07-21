# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import trimesh

from isaaclab_arena.relations.placement_asset import PlacementAsset
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


class DummyObject(PlacementAsset):
    """Dummy object for testing purposes without Isaac Sim dependencies."""

    def __init__(
        self,
        name: str,
        bounding_box: AxisAlignedBoundingBox,
        initial_pose: Pose | None = None,
        relations: list[RelationBase] | None = None,
        collision_mesh: trimesh.Trimesh | None = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.initial_pose = initial_pose
        self.bounding_box = bounding_box
        assert self.bounding_box is not None
        self.relations = list(relations or [])
        self._collision_mesh = collision_mesh

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        return self.bounding_box

    def get_corners_aabb(self, pos: torch.Tensor) -> torch.Tensor:
        return self.bounding_box.get_corners_at(pos)

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the collision mesh, or None to fall back to AABB."""
        return self._collision_mesh
