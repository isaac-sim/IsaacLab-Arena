# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import types

import pytest


class _FakeScene:
    def __init__(self, assets: dict[str, object], env_origins: torch.Tensor):
        self._assets = assets
        self.env_origins = env_origins

    def __getitem__(self, name: str):
        return self._assets[name]

    def keys(self):
        return self._assets.keys()


class _FakeEnv:
    def __init__(self, assets: dict[str, object], env_origins: torch.Tensor):
        self.scene = _FakeScene(assets, env_origins)
        self.num_envs = env_origins.shape[0]
        self.device = env_origins.device
        self.unwrapped = self


def test_object_state_reads_rigid_root_channels() -> None:
    from isaaclab_arena.scene.object_state import object_state

    env_origins = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    root_pose_w = torch.tensor([
        [1.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
        [2.4, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0],
    ])
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pose_w=root_pose_w,
            root_pos_w=root_pose_w[:, :3],
            root_lin_vel_w=torch.ones(2, 3),
            root_ang_vel_w=torch.full((2, 3), 2.0),
        )
    )
    state = object_state(_FakeEnv({"cube": asset}, env_origins), "cube")

    assert torch.allclose(state.position_w(), root_pose_w[:, :3])
    assert torch.allclose(state.pose_w(relative=True)[:, :3], root_pose_w[:, :3] - env_origins)
    assert torch.allclose(state.linear_velocity_w(), torch.ones(2, 3))
    assert torch.allclose(state.angular_velocity_w(), torch.full((2, 3), 2.0))
    assert state.capabilities.has_angular_velocity is True


def test_object_state_reads_deformable_centroid_channels() -> None:
    from isaaclab_arena.scene.object_state import object_state

    env_origins = torch.zeros(2, 3)
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            root_vel_w=torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]),
        )
    )
    state = object_state(_FakeEnv({"cloth": asset}, env_origins), "cloth")

    assert torch.allclose(state.position_w(), asset.data.root_pos_w)
    assert torch.allclose(state.linear_velocity_w(), asset.data.root_vel_w)
    assert torch.allclose(state.quat_w(), torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))
    assert torch.allclose(state.angular_velocity_w(required=False), torch.zeros(2, 3))
    assert state.capabilities.has_angular_velocity is False
    with pytest.raises(AttributeError, match="angular velocity"):
        state.angular_velocity_w(required=True)


def test_spatial_predicates_use_shared_object_state() -> None:
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.tasks.predicates.spatial import (
        object_is_above_height,
        object_is_below_height,
        object_moving,
        objects_in_proximity,
    )

    env = _FakeEnv(
        {
            "soft": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pos_w=torch.tensor([[0.0, 0.0, 0.2], [0.5, 0.0, -0.1]]),
                    root_vel_w=torch.tensor([[0.01, 0.0, 0.0], [0.3, 0.0, 0.0]]),
                )
            ),
            "target": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pos_w=torch.tensor([[0.03, 0.02, 0.22], [0.9, 0.0, 0.0]]),
                    root_lin_vel_w=torch.zeros(2, 3),
                )
            ),
        },
        torch.zeros(2, 3),
    )

    assert object_is_above_height(env, "soft", surface_height=0.0).tolist() == [True, False]
    assert object_is_below_height(env, "soft", minimum_height=0.0).tolist() == [False, True]
    assert object_moving(env, "soft", velocity_threshold=0.1).tolist() == [False, True]
    assert objects_in_proximity(
        env,
        object_cfg=SceneEntityCfg("soft"),
        target_object_cfg=SceneEntityCfg("target"),
        max_x_separation=0.1,
        max_y_separation=0.1,
        max_z_separation=0.1,
        velocity_threshold=0.1,
    ).tolist() == [True, False]
