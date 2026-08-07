# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_arena.scene.object_geometry import object_geometry
from isaaclab_arena.scene.object_state import object_state


def object_position_in_world_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Observation the position of the object in the world frame."""
    return object_state(env, asset_cfg.name).position_w()


def object_position_in_frame(
    env: ManagerBasedRLEnv,
    root_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    root_frame: RigidObject = env.scene[root_frame_cfg.name]
    object_pos_w = object_state(env, object_cfg.name).position_w()
    object_pos_b, _ = subtract_frame_transforms(
        wp.to_torch(root_frame.data.root_pos_w), wp.to_torch(root_frame.data.root_quat_w), object_pos_w
    )
    return object_pos_b


class ObjectSampledPointsInFrame(ManagerTermBase):
    """Sample representative object points in another scene frame."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.root_frame_cfg: SceneEntityCfg = cfg.params.get("root_frame_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 20)

        points = object_geometry(env, self.object_cfg.name).key_points_w()
        self.num_available_points = points.shape[1]
        self.point_ids = torch.empty(env.num_envs, self.num_points, dtype=torch.long, device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample representative points for selected environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        if self.num_points <= self.num_available_points:
            self.point_ids[env_ids] = (
                torch.rand((num_envs, self.num_available_points), device=self.device)
                .topk(self.num_points, dim=1)
                .indices
            )
        else:
            self.point_ids[env_ids] = torch.randint(
                self.num_available_points,
                (num_envs, self.num_points),
                device=self.device,
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        root_frame_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_points: int = 20,
    ) -> torch.Tensor:
        """Return flattened sampled object points in ``root_frame_cfg``."""
        if object_cfg.name != self.object_cfg.name or root_frame_cfg.name != self.root_frame_cfg.name:
            raise ValueError("ObjectSampledPointsInFrame must be called with the cfg names used at construction.")
        if num_points != self.num_points:
            raise ValueError(f"Requested {num_points} points, but this term was initialized with {self.num_points}.")

        points_w = object_geometry(env, object_cfg.name).key_points_w()
        sampled_points_w = points_w.gather(1, self.point_ids.unsqueeze(-1).expand(-1, -1, 3))
        root_frame: RigidObject = env.scene[root_frame_cfg.name]
        root_pos_w = wp.to_torch(root_frame.data.root_pos_w).unsqueeze(1).expand(-1, num_points, -1)
        root_quat_w = wp.to_torch(root_frame.data.root_quat_w).unsqueeze(1).expand(-1, num_points, -1)
        sampled_points_b, _ = subtract_frame_transforms(
            root_pos_w.reshape(-1, 3),
            root_quat_w.reshape(-1, 4),
            sampled_points_w.reshape(-1, 3),
        )
        return sampled_points_b.view(env.num_envs, -1)
