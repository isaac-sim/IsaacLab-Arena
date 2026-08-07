# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class ObjectStateCapabilities:
    """Runtime state channels exposed by a scene object."""

    has_position: bool
    has_orientation: bool
    has_linear_velocity: bool
    has_angular_velocity: bool


def get_env(env):
    """Resolve to the unwrapped manager-based env regardless of wrapper depth."""
    seen = set()
    while hasattr(env, "unwrapped") and env.unwrapped is not env and id(env) not in seen:
        seen.add(id(env))
        env = env.unwrapped
    return env


def object_state(env, name: str) -> ObjectState:
    """Return the unified runtime state view for a scene object."""
    return ObjectState(get_env(env), name)


class ObjectState:
    """Runtime state view for Arena objects backed by different Isaac Lab asset classes."""

    def __init__(self, env, name: str):
        self.env = env
        self.name = name
        assert name in env.scene.keys(), f"Asset {name} not found in scene"
        self.entity = env.scene[name]

    @property
    def data(self):
        return getattr(self.entity, "data", None)

    @property
    def capabilities(self) -> ObjectStateCapabilities:
        return ObjectStateCapabilities(
            has_position=self._has_position(),
            has_orientation=self._has_orientation(),
            has_linear_velocity=self._has_linear_velocity(),
            has_angular_velocity=self._has_angular_velocity(),
        )

    def position_w(self) -> torch.Tensor:
        """Return object position in the world frame, shape ``(num_envs, 3)``."""
        data = self.data
        if data is not None and hasattr(data, "root_pos_w"):
            return _to_torch(data.root_pos_w)
        if data is not None and hasattr(data, "root_pose_w"):
            return _to_torch(data.root_pose_w)[:, :3]
        if hasattr(self.entity, "get_world_poses"):
            pos_w, _ = self.entity.get_world_poses()
            return _to_torch(pos_w)
        raise AttributeError(f"Scene object '{self.name}' does not expose a world position.")

    def quat_w(self, required: bool = False) -> torch.Tensor:
        """Return object orientation in the world frame, shape ``(num_envs, 4)``.

        Objects without an orientation, such as deformable centroids, return identity quaternions
        unless ``required`` is True.
        """
        data = self.data
        if data is not None and hasattr(data, "root_quat_w"):
            return _to_torch(data.root_quat_w)
        if data is not None and hasattr(data, "root_pose_w"):
            return _to_torch(data.root_pose_w)[:, 3:7]
        if hasattr(self.entity, "get_world_poses"):
            _, quat_w = self.entity.get_world_poses()
            return _to_torch(quat_w)
        if required:
            raise AttributeError(f"Scene object '{self.name}' does not expose an orientation.")
        quat = torch.zeros((self.env.num_envs, 4), device=self.env.device)
        quat[:, 3] = 1.0
        return quat

    def pose_w(self, relative: bool = False) -> torch.Tensor:
        """Return object pose in world or env-relative frame, shape ``(num_envs, 7)``."""
        data = self.data
        if data is not None and hasattr(data, "root_pose_w"):
            pose = _to_torch(data.root_pose_w).clone()
        else:
            pose = torch.cat([self.position_w(), self.quat_w()], dim=-1)
        if relative:
            pose[:, :3] -= self.env.scene.env_origins
        return pose

    def linear_velocity_w(self, required: bool = True) -> torch.Tensor:
        """Return object linear velocity in the world frame, shape ``(num_envs, 3)``."""
        data = self.data
        if data is not None and hasattr(data, "root_lin_vel_w"):
            return _to_torch(data.root_lin_vel_w)
        if data is not None and hasattr(data, "root_vel_w"):
            root_vel_w = _to_torch(data.root_vel_w)
            return root_vel_w[:, :3] if root_vel_w.shape[-1] >= 6 else root_vel_w
        if data is not None and hasattr(data, "root_link_vel_w"):
            return _to_torch(data.root_link_vel_w)[:, :3]
        if required:
            raise AttributeError(f"Scene object '{self.name}' does not expose a linear velocity.")
        return torch.zeros_like(self.position_w())

    def angular_velocity_w(self, required: bool = True) -> torch.Tensor:
        """Return object angular velocity in the world frame, shape ``(num_envs, 3)``."""
        data = self.data
        if data is not None and hasattr(data, "root_ang_vel_w"):
            return _to_torch(data.root_ang_vel_w)
        if data is not None and hasattr(data, "root_link_vel_w"):
            root_link_vel_w = _to_torch(data.root_link_vel_w)
            if root_link_vel_w.shape[-1] >= 6:
                return root_link_vel_w[:, 3:6]
        if required:
            raise AttributeError(f"Scene object '{self.name}' does not expose an angular velocity.")
        return torch.zeros_like(self.position_w())

    def _has_position(self) -> bool:
        data = self.data
        return (data is not None and (hasattr(data, "root_pos_w") or hasattr(data, "root_pose_w"))) or hasattr(
            self.entity, "get_world_poses"
        )

    def _has_orientation(self) -> bool:
        data = self.data
        return (data is not None and (hasattr(data, "root_quat_w") or hasattr(data, "root_pose_w"))) or hasattr(
            self.entity, "get_world_poses"
        )

    def _has_linear_velocity(self) -> bool:
        data = self.data
        return data is not None and (
            hasattr(data, "root_lin_vel_w") or hasattr(data, "root_vel_w") or hasattr(data, "root_link_vel_w")
        )

    def _has_angular_velocity(self) -> bool:
        data = self.data
        return data is not None and (
            hasattr(data, "root_ang_vel_w")
            or (hasattr(data, "root_link_vel_w") and _to_torch(data.root_link_vel_w).shape[-1] >= 6)
        )


def _to_torch(value) -> torch.Tensor:
    """Convert torch, Warp, and Isaac Lab tensor wrappers to ``torch.Tensor``."""
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    return wp.to_torch(value)
