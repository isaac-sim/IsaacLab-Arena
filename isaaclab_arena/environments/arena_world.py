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
    """Provide name-based live scene queries and cache derived geometry."""

    def __init__(self, scene: InteractiveScene):
        self._scene: InteractiveScene | None = scene
        self._aabbs_in_entity_frame: dict[str, AxisAlignedBoundingBox] = {}
        self._asset_base_cfg_pose_readers: dict[str, entity_access.AssetBaseCfgPoseReader] = {}

    def get_pose_w(self, entity_name: str) -> torch.Tensor:
        """Return ``T_W_E``, mapping the named entity frame ``E`` into world frame ``W``.

        Poses have shape ``(num_envs, 7)`` and quaternion order ``(x, y, z, w)``.
        """
        scene = self._get_scene()
        if entity_name in scene.rigid_objects:
            T_W_E = scene.rigid_objects[entity_name].data.root_pose_w.torch
            assert T_W_E.shape == (
                scene.num_envs,
                7,
            ), f"Rigid object '{entity_name}' returned pose shape {tuple(T_W_E.shape)}; expected ({scene.num_envs}, 7)."
            return T_W_E

        assert (
            entity_name in scene.extras
        ), f"Scene entity '{entity_name}' must be a rigid object or an AssetBaseCfg scene entry."
        if entity_name not in self._asset_base_cfg_pose_readers:
            self._asset_base_cfg_pose_readers[entity_name] = entity_access.AssetBaseCfgPoseReader(scene, entity_name)
        return self._asset_base_cfg_pose_readers[entity_name].get_pose_w()

    def get_linear_velocity_w(self, entity_name: str) -> torch.Tensor:
        """Return a rigid object's current world-frame linear velocity."""
        scene = self._get_scene()
        assert entity_name in scene.rigid_objects, f"Scene entity '{entity_name}' must be a rigid object."
        linear_velocity_w = scene.rigid_objects[entity_name].data.root_lin_vel_w.torch
        assert linear_velocity_w.shape == (scene.num_envs, 3), (
            f"Rigid object '{entity_name}' returned linear velocity shape {tuple(linear_velocity_w.shape)}; "
            f"expected ({scene.num_envs}, 3)."
        )
        return linear_velocity_w

    def get_aabb_in_entity_frame(self, entity_name: str) -> AxisAlignedBoundingBox:
        """Return cached geometry bounds expressed in the named entity frame ``E``."""
        scene = self._get_scene()
        if entity_name not in self._aabbs_in_entity_frame:
            aabb_E = entity_access.compute_spawned_geometry_bounds_in_entity_frame(scene, entity_name)
            self._aabbs_in_entity_frame[entity_name] = aabb_E
        return self._aabbs_in_entity_frame[entity_name]

    def close(self) -> None:
        """Release cached geometry, pose readers, and the live scene reference."""
        self._aabbs_in_entity_frame.clear()
        self._asset_base_cfg_pose_readers.clear()
        self._scene = None

    def _get_scene(self) -> InteractiveScene:
        """Return the live scene while this world is open."""
        assert self._scene is not None, "ArenaWorld is closed."
        return self._scene
