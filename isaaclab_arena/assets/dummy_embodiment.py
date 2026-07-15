# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight placement embodiment for solver tests."""

from __future__ import annotations

import trimesh

from isaaclab_arena.relations.placement_entity import PlacementEntity
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose


class DummyEmbodiment(PlacementEntity):
    """Embodiment geometry without simulator dependencies."""

    def __init__(
        self,
        name: str,
        bounding_box: AxisAlignedBoundingBox,
        initial_pose: Pose | None = None,
        collision_mesh: trimesh.Trimesh | None = None,
    ) -> None:
        super().__init__(name=name, tags=["embodiment"])
        self.initial_pose = initial_pose
        self.bounding_box = bounding_box
        self.collision_mesh = collision_mesh

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return root-relative bounds."""
        return self.bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return bounds transformed by the root pose."""
        if self.initial_pose is None:
            return self.bounding_box
        quarters = quaternion_to_90_deg_z_quarters(self.initial_pose.rotation_xyzw)
        return self.bounding_box.rotated_90_around_z(quarters).translated(self.initial_pose.position_xyz)

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the configured collision mesh."""
        return self.collision_mesh

    def supports_per_env_initial_pose(self) -> bool:
        """Return False because the dummy stores one root pose."""
        return False
