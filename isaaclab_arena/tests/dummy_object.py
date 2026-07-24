# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg


class DummyObject(PlaceableAsset):
    """Dummy object for testing purposes without Isaac Sim dependencies.

    Mirrors the real object pose lifecycle: setting an initial pose builds a
    root-pose reset event, so the solver's ``has_pose_reset_event`` invariant holds.
    """

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

    def _build_reset_event(self) -> EventTermCfg | None:
        from isaaclab.managers import EventTermCfg

        from isaaclab_arena.terms.events import reset_placement_asset_pose, reset_placement_asset_pose_per_env

        if self.initial_pose is None:
            return None
        if isinstance(self.initial_pose, PosePerEnv):
            return EventTermCfg(
                func=reset_placement_asset_pose_per_env,
                mode="reset",
                params={"write_pose_list": [self.layout_pose_to_scene_writes(p) for p in self.initial_pose.poses]},
            )
        return EventTermCfg(
            func=reset_placement_asset_pose,
            mode="reset",
            params={"scene_writes": self.layout_pose_to_scene_writes(self.initial_pose)},
        )

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
