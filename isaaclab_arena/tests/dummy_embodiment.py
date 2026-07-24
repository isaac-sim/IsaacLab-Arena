# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight placement embodiment for solver tests."""

from __future__ import annotations

import trimesh
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg


class DummyEmbodiment(PlaceableAsset):
    """Embodiment geometry without simulator dependencies.

    Mirrors the real embodiment pose lifecycle: setting an initial pose builds a
    root-pose reset event, so the solver's ``has_pose_reset_event`` invariant holds.
    """

    def __init__(
        self,
        name: str,
        bounding_box: AxisAlignedBoundingBox,
        initial_pose: Pose | None = None,
        collision_mesh: trimesh.Trimesh | None = None,
        scene_name: str | None = None,
    ) -> None:
        super().__init__(name=name, tags=["embodiment"])
        self.initial_pose = initial_pose
        self.bounding_box = bounding_box
        self.collision_mesh = collision_mesh
        self.scene_name = name if scene_name is None else scene_name

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
        """Return root-relative bounds."""
        return self.bounding_box

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the configured collision mesh."""
        return self.collision_mesh

    def get_scene_key(self) -> str:
        """Return the configured scene key."""
        return self.scene_name
