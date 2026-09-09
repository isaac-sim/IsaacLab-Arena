# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Query live Arena scene state and cache derived geometry.

Arena transforms use target-source notation: T_A_B maps points from frame B
into frame A. W is the simulation world, and F is the queried root-link or prim
frame. E is each Isaac Lab local environment frame, aligned with W and located
at the corresponding row of scene.env_origins. Pose method suffixes _w and _e
indicate whether a pose is expressed in W or E.
"""

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveScene

import isaaclab_arena.environments.arena_world_scene_access as scene_access
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class ArenaWorld:
    """Provide name-based pose, velocity, and geometry queries."""

    def __init__(self, scene: InteractiveScene):
        self._scene = scene
        self._aabbs_in_local_frame_cache: dict[str, AxisAlignedBoundingBox] = {}
        self._scene_extra_pose_reader_cache: dict[str, scene_access.SceneExtraPoseReader] = {}

    def get_pose_w(self, scene_key: str) -> torch.Tensor:
        """Return the world-frame pose of a rigid-object root link, articulation root link, or scene extra.

        The tensor has shape (num_envs, 7), with each pose ordered as
        (x, y, z, qx, qy, qz, qw).
        """
        scene = self._scene
        # Rigid objects and articulations expose their live root-link poses directly. Scene
        # extras are plain cloned prims, so their live post-clone poses require a FrameView-backed reader.
        if scene_key in scene.rigid_objects:
            T_W_F = scene.rigid_objects[scene_key].data.root_pose_w.torch
        elif scene_key in scene.articulations:
            T_W_F = scene.articulations[scene_key].data.root_pose_w.torch
        else:
            assert scene_key in scene.extras, (
                "ArenaWorld pose queries require a scene key registered in InteractiveScene.rigid_objects, "
                "InteractiveScene.articulations, or InteractiveScene.extras; "
                f"'{scene_key}' is registered in none of them."
            )
            pose_reader = self._get_scene_extra_pose_reader(scene, scene_key)
            T_W_F = pose_reader.get_pose_w()

        assert T_W_F.shape == (
            scene.num_envs,
            7,
        ), f"Pose for scene key '{scene_key}' has shape {tuple(T_W_F.shape)}; expected ({scene.num_envs}, 7)."
        return T_W_F

    def get_pose_e(self, scene_key: str) -> torch.Tensor:
        """Return poses relative to their respective environment origins.

        The returned tensor is a copy with shape (num_envs, 7), with each pose
        ordered as (x, y, z, qx, qy, qz, qw).
        """
        T_E_F = self.get_pose_w(scene_key).clone()
        T_E_F[:, :3] -= self._scene.env_origins
        return T_E_F

    def get_root_linear_velocity_w(self, scene_key: str) -> torch.Tensor:
        """Return the world-frame root linear velocity of a rigid object or articulation.

        The tensor has shape (num_envs, 3).
        """
        scene = self._scene
        if scene_key in scene.rigid_objects:
            root_asset = scene.rigid_objects[scene_key]
        else:
            assert scene_key in scene.articulations, (
                "ArenaWorld root velocity queries require a scene key registered in InteractiveScene.rigid_objects "
                f"or InteractiveScene.articulations; '{scene_key}' is registered in neither."
            )
            root_asset = scene.articulations[scene_key]
        root_linear_velocity_w = root_asset.data.root_lin_vel_w.torch
        assert root_linear_velocity_w.shape == (scene.num_envs, 3), (
            f"Scene key '{scene_key}' returned root linear velocity shape "
            f"{tuple(root_linear_velocity_w.shape)}; expected ({scene.num_envs}, 3)."
        )
        return root_linear_velocity_w

    def get_root_angular_velocity_w(self, scene_key: str) -> torch.Tensor:
        """Return the world-frame root angular velocity of a rigid object or articulation.

        The tensor has shape (num_envs, 3).
        """
        scene = self._scene
        if scene_key in scene.rigid_objects:
            root_asset = scene.rigid_objects[scene_key]
        else:
            assert scene_key in scene.articulations, (
                "ArenaWorld root velocity queries require a scene key registered in InteractiveScene.rigid_objects "
                f"or InteractiveScene.articulations; '{scene_key}' is registered in neither."
            )
            root_asset = scene.articulations[scene_key]
        root_angular_velocity_w = root_asset.data.root_ang_vel_w.torch
        assert root_angular_velocity_w.shape == (scene.num_envs, 3), (
            f"Scene key '{scene_key}' returned root angular velocity shape "
            f"{tuple(root_angular_velocity_w.shape)}; expected ({scene.num_envs}, 3)."
        )
        return root_angular_velocity_w

    def get_aabb_in_local_frame(self, scene_key: str) -> AxisAlignedBoundingBox:
        """Return cached rigid-object or scene-extra geometry bounds in local frame F.

        The cache assumes descendants remain fixed relative to F.
        """
        scene = self._scene
        if scene_key not in self._aabbs_in_local_frame_cache:
            aabb_F = scene_access.compute_spawned_geometry_bounds_in_local_frame(scene, scene_key)
            self._aabbs_in_local_frame_cache[scene_key] = aabb_F
        return self._aabbs_in_local_frame_cache[scene_key]

    def _get_scene_extra_pose_reader(
        self,
        scene: InteractiveScene,
        scene_extra_key: str,
    ) -> scene_access.SceneExtraPoseReader:
        """Return the cached live-pose reader for a scene extra."""
        if scene_extra_key not in self._scene_extra_pose_reader_cache:
            self._scene_extra_pose_reader_cache[scene_extra_key] = scene_access.SceneExtraPoseReader(
                scene, scene_extra_key
            )
        return self._scene_extra_pose_reader_cache[scene_extra_key]
