# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded per-episode point clouds built from egocentric optical-axis depth."""

import torch

from isaaclab.utils.math import transform_points, unproject_depth

from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
from isaaclab_arena_environments.spatial_memory.geometry import inverse_transform_points
from isaaclab_arena_environments.spatial_memory.observations import ONLINE_MAP_MAX_POINTS

VOXEL_SIZE_M = 0.03
"""Edge length of a voxel representing an observed surface."""


class DepthCloudProvider:
    """Accumulate occupied voxels in the environment frame E and query them in the robot frame R."""

    def __init__(self, env, camera_name: str, far_m: float):
        self.camera_name = camera_name
        self.far_m = far_m
        self.voxels_E = [torch.empty((0, 3), dtype=torch.int64, device=env.device) for _ in range(env.num_envs)]

    def reset(self, env, env_ids):
        """Clear only episodes that reset."""
        for i in env_ids.tolist():
            self.voxels_E[i] = self.voxels_E[i][:0]

    def query(self, env):
        """Fuse valid pixels and return voxel centers and a binary validity channel."""
        camera = env.scene[self.camera_name]
        data = camera.data
        depth = data.output["distance_to_image_plane"]
        depth = depth.torch if hasattr(depth, "torch") else depth
        stride = 8
        depth = depth[:, ::stride, ::stride]
        intrinsics = data.intrinsic_matrices.torch.clone()
        intrinsics[:, :2] /= stride
        points_C = unproject_depth(depth, intrinsics)
        valid = (points_C[..., 2] > 0) & (points_C[..., 2] < self.far_m)
        episode = get_episode_state(env)
        # A reset can return the renderer's last frame from the preceding layout.
        # Wait for the first control step before fusing measurements in a new episode.
        valid &= env.episode_length_buf[:, None] > 0
        if episode.logic.withholds_camera_frames:
            valid &= ~episode.logic.camera_blocked(env, episode)[:, None]
        points_E = transform_points(points_C, data.pos_w.torch - env.scene.env_origins, data.quat_w_ros.torch)
        T_E_R = env.scene["robot"].data.root_pose_w.torch.clone()
        T_E_R[:, :3] -= env.scene.env_origins
        output = points_E.new_zeros((env.num_envs, ONLINE_MAP_MAX_POINTS, 4))
        for i, previous in enumerate(self.voxels_E):
            observed = (points_E[i, valid[i]] / VOXEL_SIZE_M).floor().long()
            voxels = torch.unique(torch.cat((previous, observed)), dim=0) if len(observed) else previous
            if len(voxels) > ONLINE_MAP_MAX_POINTS:
                # A stable spatial hash keeps repeat observations from evicting older, now hidden surfaces.
                x, y, z = voxels.unbind(dim=1)
                priority = (x * 73856093 ^ y * 19349663 ^ z * 83492791) % 2147483647
                voxels = voxels[priority.argsort(stable=True)[:ONLINE_MAP_MAX_POINTS]]
            self.voxels_E[i] = voxels
            if not len(voxels):
                continue
            centers_E = (voxels.to(points_E.dtype) + 0.5) * VOXEL_SIZE_M
            output[i, : len(voxels), :3] = inverse_transform_points(T_E_R[i], centers_E)
            output[i, : len(voxels), 3] = 1
        return output
