# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Query live Arena scene state and cache derived geometry."""

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveScene

import isaaclab_arena.environments.arena_world_entity_access as entity_access
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class ArenaWorld:
    """Provide name-based runtime access to rigid objects and scene extras.

    Poses are read live for both entity types. Rigid-object root linear velocities are also read live,
    while spawned geometry is cached for the environment lifetime.
    """

    def __init__(self, scene: InteractiveScene):
        self._scene: InteractiveScene | None = scene
        self._aabbs_in_entity_frame_cache: dict[str, AxisAlignedBoundingBox] = {}
        self._scene_extra_pose_reader_cache: dict[str, entity_access.SceneExtraPoseReader] = {}

    def get_pose_w(self, entity_name: str) -> torch.Tensor:
        """Return ``T_W_E``, mapping the named entity frame ``E`` into world frame ``W``.

        Poses have shape ``(num_envs, 7)`` and quaternion order ``(x, y, z, w)``.
        """
        scene = self._get_scene()
        is_rigid_object = entity_name in scene.rigid_objects
        is_scene_extra = entity_name in scene.extras
        assert is_rigid_object or is_scene_extra, (
            "ArenaWorld pose queries support only entities registered in InteractiveScene.rigid_objects or "
            f"InteractiveScene.extras; '{entity_name}' is registered in neither."
        )

        # Rigid objects expose live root state directly. Scene extras are plain cloned prims,
        # so their live post-clone poses require a FrameView-backed reader.
        if is_rigid_object:
            T_W_E = scene.rigid_objects[entity_name].data.root_pose_w.torch
        else:
            pose_reader = self._get_scene_extra_pose_reader(scene, entity_name)
            T_W_E = pose_reader.get_pose_w()

        assert T_W_E.shape == (
            scene.num_envs,
            7,
        ), f"Scene entity '{entity_name}' returned pose shape {tuple(T_W_E.shape)}; expected ({scene.num_envs}, 7)."
        return T_W_E

    def get_root_linear_velocity_w(self, entity_name: str) -> torch.Tensor:
        """Return a rigid object's current world-frame root linear velocity."""
        scene = self._get_scene()
        assert entity_name in scene.rigid_objects, f"Scene entity '{entity_name}' must be a rigid object."
        root_linear_velocity_w = scene.rigid_objects[entity_name].data.root_lin_vel_w.torch
        assert root_linear_velocity_w.shape == (scene.num_envs, 3), (
            f"Rigid object '{entity_name}' returned root linear velocity shape {tuple(root_linear_velocity_w.shape)}; "
            f"expected ({scene.num_envs}, 3)."
        )
        return root_linear_velocity_w

    def get_aabb_in_entity_frame(self, entity_name: str) -> AxisAlignedBoundingBox:
        """Return cached geometry bounds expressed in the named entity frame ``E``."""
        scene = self._get_scene()
        if entity_name not in self._aabbs_in_entity_frame_cache:
            aabb_E = entity_access.compute_spawned_geometry_bounds_in_entity_frame(scene, entity_name)
            self._aabbs_in_entity_frame_cache[entity_name] = aabb_E
        return self._aabbs_in_entity_frame_cache[entity_name]

    def close(self) -> None:
        """Release cached geometry, pose readers, and the live scene reference."""
        self._aabbs_in_entity_frame_cache.clear()
        self._scene_extra_pose_reader_cache.clear()
        self._scene = None

    def _get_scene_extra_pose_reader(
        self,
        scene: InteractiveScene,
        entity_name: str,
    ) -> entity_access.SceneExtraPoseReader:
        """Return the cached live-pose reader for a scene extra."""
        if entity_name not in self._scene_extra_pose_reader_cache:
            self._scene_extra_pose_reader_cache[entity_name] = entity_access.SceneExtraPoseReader(scene, entity_name)
        return self._scene_extra_pose_reader_cache[entity_name]

    def _get_scene(self) -> InteractiveScene:
        """Return the live scene while this world is open."""
        assert self._scene is not None, "ArenaWorld is closed."
        return self._scene
