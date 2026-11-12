# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Protocol for objects with queryable collision geometry."""

from __future__ import annotations

import trimesh
from typing import Protocol

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange


class CollisionObject(Protocol):
    """Object the collision solver can query for pose, bounds, and mesh."""

    name: str
    collision_mode: CollisionMode | None
    repair_collision_mesh_non_watertight: bool

    @property
    def is_anchor(self) -> bool:
        """Return whether the object is a fixed anchor that the solver does not move."""
        raise NotImplementedError

    def get_initial_pose(self) -> Pose | PoseRange | PosePerEnv | None:
        """Return the object's initial pose, pose range or per-env poses, if any."""
        raise NotImplementedError

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the object's axis-aligned bounding box relative to its origin."""
        raise NotImplementedError

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the object's axis-aligned bounding box in the world frame."""
        raise NotImplementedError

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the object's collision mesh, or None if it has none."""
        raise NotImplementedError
