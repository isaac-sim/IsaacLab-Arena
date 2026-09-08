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
from isaaclab.utils.math import quat_apply

import isaaclab_arena.environments.arena_world_scene_access as scene_access
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class ArenaWorld:
    """Provide name-based pose, velocity, and geometry queries."""

    def __init__(self, scene: InteractiveScene):
        self._scene = scene
        self._aabbs_in_local_frame_cache: dict[str, AxisAlignedBoundingBox] = {}
        self._scene_extra_pose_reader_cache: dict[str, scene_access.SceneExtraPoseReader] = {}

    def get_pose_w(self, scene_key: str) -> torch.Tensor:
        """Return the world-frame pose of a scene entity root link, prim, or deformable aggregate.

        The tensor has shape (num_envs, 7), with each pose ordered as
        (x, y, z, qx, qy, qz, qw).
        """
        scene = self._scene
        is_rigid_object = scene_key in scene.rigid_objects
        is_articulation = scene_key in scene.articulations
        is_deformable_object = scene_key in scene.deformable_objects
        is_scene_extra = scene_key in scene.extras
        assert is_rigid_object or is_articulation or is_deformable_object or is_scene_extra, (
            "ArenaWorld pose queries require a scene key registered in InteractiveScene.rigid_objects, "
            "InteractiveScene.articulations, InteractiveScene.deformable_objects, or InteractiveScene.extras; "
            f"'{scene_key}' is registered in none of them."
        )

        # Rigid objects and articulations expose their live root-link poses directly. Deformables
        # expose aggregate position with identity rotation. Scene extras are plain cloned prims,
        # so their live post-clone poses require a FrameView-backed reader.
        if is_rigid_object:
            T_W_F = scene.rigid_objects[scene_key].data.root_pose_w.torch
        elif is_articulation:
            T_W_F = scene.articulations[scene_key].data.root_pose_w.torch
        elif is_deformable_object:
            root_pos_w = scene.deformable_objects[scene_key].data.root_pos_w.torch
            # Deformable object root does not have a rotation, use an identity as dummy value.
            identity_quat = root_pos_w.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(scene.num_envs, 4)
            T_W_F = torch.cat((root_pos_w, identity_quat), dim=-1)
        else:
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

    def get_position_w(self, scene_key: str) -> torch.Tensor:
        """Return the world-frame position with shape (num_envs, 3)."""
        return self.get_pose_w(scene_key)[:, :3]

    def get_root_linear_velocity_w(self, scene_key: str) -> torch.Tensor:
        """Return an entity's current world-frame root linear velocity.

        The tensor has shape (num_envs, 3).
        """
        scene = self._scene
        if scene_key in scene.rigid_objects:
            root_linear_velocity_w = scene.rigid_objects[scene_key].data.root_lin_vel_w.torch
        elif scene_key in scene.articulations:
            root_linear_velocity_w = scene.articulations[scene_key].data.root_lin_vel_w.torch
        else:
            assert (
                scene_key in scene.deformable_objects
            ), f"'{scene_key}' must name a rigid object, articulation, or deformable object."
            root_linear_velocity_w = scene.deformable_objects[scene_key].data.root_vel_w.torch
        assert root_linear_velocity_w.shape == (scene.num_envs, 3), (
            f"Scene key '{scene_key}' returned root linear velocity shape "
            f"{tuple(root_linear_velocity_w.shape)}; expected ({scene.num_envs}, 3)."
        )
        return root_linear_velocity_w

    def get_root_angular_velocity_w(self, scene_key: str) -> torch.Tensor | None:
        """Return a rigid root's world-frame angular velocity, or None for deformables.

        Args:
            scene_key: Scene entity name.

        Returns:
            Angular velocity with shape (num_envs, 3) for rigid objects and articulations,
            or None when ``scene_key`` names a deformable object.
        """
        scene = self._scene
        if scene_key in scene.deformable_objects:
            return None
        assert (
            scene_key in scene.rigid_objects or scene_key in scene.articulations
        ), f"'{scene_key}' must name a rigid object, articulation, or deformable object."
        asset = scene.rigid_objects.get(scene_key, scene.articulations.get(scene_key))
        root_angular_velocity_w = asset.data.root_ang_vel_w.torch
        assert root_angular_velocity_w.shape == (scene.num_envs, 3), (
            f"Scene key '{scene_key}' returned root angular velocity shape "
            f"{tuple(root_angular_velocity_w.shape)}; expected ({scene.num_envs}, 3)."
        )
        return root_angular_velocity_w

    def get_max_point_speed_w(self, scene_key: str) -> torch.Tensor:
        """Return maximum nodal speed for a deformable, or root linear speed for a rigid object.

        The tensor has shape (num_envs,).
        """
        scene = self._scene
        if scene_key in scene.deformable_objects:
            nodal_velocity_w = scene.deformable_objects[scene_key].data.nodal_vel_w.torch
            max_point_speed_w = torch.linalg.vector_norm(nodal_velocity_w, dim=-1).amax(dim=1)
        else:
            # Using root linear velocity for rigid objects. This is an approximation as it does
            # not account for angular velocity.
            max_point_speed_w = torch.linalg.vector_norm(self.get_root_linear_velocity_w(scene_key), dim=-1)
        assert max_point_speed_w.shape == (scene.num_envs,), (
            f"Scene object '{scene_key}' returned max point speed shape {tuple(max_point_speed_w.shape)}; "
            f"expected ({scene.num_envs},)."
        )
        return max_point_speed_w

    def get_vertices_pos_w(self, scene_key: str) -> torch.Tensor:
        """Return deformable nodes or approximate geometry vertices in world frame ``W``.

        The tensor has shape ``(num_envs, num_vertices, 3)``.
        """
        scene = self._scene
        if scene_key in scene.deformable_objects:
            vertices_pos_w = scene.deformable_objects[scene_key].data.nodal_pos_w.torch
        else:
            # TODO(qianl, 2026-09-08): Return actual vertices once the rigid mesh cache is added.
            vertices_pos_F = self.get_aabb_in_local_frame(scene_key).get_corners_at()
            if vertices_pos_F.shape[0] == 1 and scene.num_envs > 1:
                vertices_pos_F = vertices_pos_F.expand(scene.num_envs, -1, -1)
            T_W_F = self.get_pose_w(scene_key)
            t_W_F, q_W_F = T_W_F[:, :3], T_W_F[:, 3:]
            q_W_F = q_W_F[:, None, :].expand(-1, vertices_pos_F.shape[1], -1)
            vertices_pos_w = quat_apply(q_W_F, vertices_pos_F) + t_W_F[:, None, :]
        assert vertices_pos_w.shape[0] == scene.num_envs and vertices_pos_w.shape[2] == 3, (
            f"Scene entity '{scene_key}' returned vertices shape {tuple(vertices_pos_w.shape)}; "
            f"expected ({scene.num_envs}, num_vertices, 3)."
        )
        return vertices_pos_w

    def get_min_height_w(self, scene_key: str) -> torch.Tensor:
        """Return the lowest world-frame Z for a deformable or rigid scene entity.

        The tensor has shape ``(num_envs,)``.
        """
        scene = self._scene
        min_height_w = self.get_vertices_pos_w(scene_key)[..., 2].amin(dim=1)
        assert min_height_w.shape == (scene.num_envs,), (
            f"Scene entity '{scene_key}' returned min height shape {tuple(min_height_w.shape)}; "
            f"expected ({scene.num_envs},)."
        )
        return min_height_w

    def get_aabb_in_local_frame(self, scene_key: str) -> AxisAlignedBoundingBox:
        """Return cached rigid-object or scene-extra geometry bounds in local frame F.

        The cache assumes descendants remain fixed relative to F.
        """
        scene = self._scene
        if scene_key not in self._aabbs_in_local_frame_cache:
            aabb_F = scene_access.compute_spawned_geometry_bounds_in_local_frame(scene, scene_key)
            self._aabbs_in_local_frame_cache[scene_key] = aabb_F
        return self._aabbs_in_local_frame_cache[scene_key]

    def get_aabb_w(self, scene_key: str) -> AxisAlignedBoundingBox:
        """Return cached local bounds transformed to the entity's current world pose."""
        T_W_F = self.get_pose_w(scene_key)
        t_W_F, q_W_F = T_W_F[:, :3], T_W_F[:, 3:]
        bounds_F = self.get_aabb_in_local_frame(scene_key)
        return bounds_F.rotated_by_quat(q_W_F).translated(t_W_F)

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
