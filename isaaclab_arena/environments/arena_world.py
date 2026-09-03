# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Query live Arena scene state and cache derived geometry.

Arena pose and transform names use target-source notation: T_A_B maps points
from frame B into frame A. Here W is the simulation world, F is the frame of
the rigid object or scene extra selected by a scene key, and geometry helpers
use P for a USD prim's local frame. For a rigid object, Isaac Lab's root_pose_w
supplies the value represented here as T_W_F.
"""

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveScene

import isaaclab_arena.environments.arena_world_scene_access as scene_access
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class ArenaWorld:
    """Provide name-based runtime access to rigid objects and scene extras.

    Poses are read live for both supported scene categories, along with root linear velocities for rigid objects.
    Local-frame geometry bounds are computed lazily from the cloned prim hierarchy and cached for the environment
    lifetime. They remain valid under whole-subtree motion, but not when descendants move relative to frame F.
    A moving part must therefore have its own supported scene key.
    """

    def __init__(self, scene: InteractiveScene):
        self._scene = scene
        self._aabbs_in_local_frame_cache: dict[str, AxisAlignedBoundingBox] = {}
        self._scene_extra_pose_reader_cache: dict[str, scene_access.SceneExtraPoseReader] = {}

    def get_pose_w(self, scene_key: str) -> torch.Tensor:
        """Return the current world-frame pose for a rigid-object or scene-extra key.

        The tensor has shape (num_envs, 7), with each pose ordered as
        (x, y, z, qx, qy, qz, qw).
        """
        scene = self._scene
        is_rigid_object = scene_key in scene.rigid_objects
        is_scene_extra = scene_key in scene.extras
        assert is_rigid_object or is_scene_extra, (
            "ArenaWorld pose queries require a scene key registered in InteractiveScene.rigid_objects or "
            f"InteractiveScene.extras; '{scene_key}' is registered in neither."
        )

        # Rigid objects expose live root state directly. Scene extras are plain cloned prims,
        # so their live post-clone poses require a FrameView-backed reader.
        if is_rigid_object:
            T_W_F = scene.rigid_objects[scene_key].data.root_pose_w.torch
        else:
            pose_reader = self._get_scene_extra_pose_reader(scene, scene_key)
            T_W_F = pose_reader.get_pose_w()

        assert T_W_F.shape == (
            scene.num_envs,
            7,
        ), f"Pose for scene key '{scene_key}' has shape {tuple(T_W_F.shape)}; expected ({scene.num_envs}, 7)."
        return T_W_F

    def get_root_linear_velocity_w(self, rigid_object_name: str) -> torch.Tensor:
        """Return a rigid object's current world-frame root linear velocity.

        The tensor has shape (num_envs, 3).
        """
        scene = self._scene
        assert rigid_object_name in scene.rigid_objects, f"'{rigid_object_name}' must name a rigid object."
        root_linear_velocity_w = scene.rigid_objects[rigid_object_name].data.root_lin_vel_w.torch
        assert root_linear_velocity_w.shape == (scene.num_envs, 3), (
            f"Rigid object '{rigid_object_name}' returned root linear velocity shape "
            f"{tuple(root_linear_velocity_w.shape)}; expected ({scene.num_envs}, 3)."
        )
        return root_linear_velocity_w

    def get_aabb_in_local_frame(self, scene_key: str) -> AxisAlignedBoundingBox:
        """Return cached geometry bounds expressed in the selected local frame F."""
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
