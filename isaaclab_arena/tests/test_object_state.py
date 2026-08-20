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
    def __init__(
        self,
        assets: dict[str, object],
        env_origins: torch.Tensor,
        arena_scene_assets: dict[str, object] | None = None,
    ):
        self.scene = _FakeScene(assets, env_origins)
        self.num_envs = env_origins.shape[0]
        self.device = env_origins.device
        self.unwrapped = self
        self.arena_scene_assets = dict(arena_scene_assets or {})


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


def test_object_geometry_reads_deformable_nodal_channels() -> None:
    from isaaclab_arena.scene.object_geometry import object_geometry

    nodal_pos_w = torch.tensor([
        [[0.0, 0.0, 0.0], [0.2, 0.1, 0.3], [0.1, -0.2, 0.1]],
        [[1.0, 0.0, 0.0], [1.2, 0.2, 0.1], [0.8, -0.1, 0.4]],
    ])
    nodal_vel_w = torch.tensor([
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
        [[0.0, 0.0, 0.3], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
    ])
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            nodal_pos_w=nodal_pos_w,
            nodal_vel_w=nodal_vel_w,
            root_pos_w=nodal_pos_w.mean(dim=1),
            root_vel_w=nodal_vel_w.mean(dim=1),
        )
    )
    geometry = object_geometry(_FakeEnv({"soft": asset}, torch.zeros(2, 3)), "soft")

    bbox = geometry.aabb_w()
    assert torch.allclose(bbox.min_point, nodal_pos_w.amin(dim=1))
    assert torch.allclose(bbox.max_point, nodal_pos_w.amax(dim=1))
    assert torch.allclose(geometry.max_point_speed(), torch.tensor([0.2, 0.4]))
    nearest = geometry.nearest_point_w(torch.tensor([[0.21, 0.1, 0.31], [0.75, -0.1, 0.4]]))
    assert torch.allclose(nearest, torch.stack([nodal_pos_w[0, 1], nodal_pos_w[1, 2]]))


def test_object_geometry_uses_static_asset_bbox_for_rigid() -> None:
    from isaaclab_arena.scene.object_geometry import object_geometry
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    root_pose_w = torch.tensor([
        [1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        [3.0, 4.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    ])
    runtime_asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pose_w=root_pose_w,
            root_pos_w=root_pose_w[:, :3],
            root_lin_vel_w=torch.zeros(2, 3),
        )
    )
    arena_asset = types.SimpleNamespace(
        get_bounding_box=lambda: AxisAlignedBoundingBox(min_point=(-0.1, -0.2, 0.0), max_point=(0.1, 0.2, 0.4))
    )
    geometry = object_geometry(
        _FakeEnv({"box": runtime_asset}, torch.zeros(2, 3), arena_scene_assets={"box": arena_asset}), "box"
    )

    bbox = geometry.aabb_w()
    assert torch.allclose(bbox.min_point, torch.tensor([[0.9, 1.8, 0.5], [2.9, 3.8, 0.5]]))
    assert torch.allclose(bbox.max_point, torch.tensor([[1.1, 2.2, 0.9], [3.1, 4.2, 0.9]]))


def test_object_geometry_uses_per_env_asset_bbox_when_available() -> None:
    from isaaclab_arena.scene.object_geometry import object_geometry
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    root_pose_w = torch.tensor([
        [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    ])
    runtime_asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pose_w=root_pose_w,
            root_pos_w=root_pose_w[:, :3],
            root_lin_vel_w=torch.zeros(2, 3),
        )
    )

    class _ArenaAsset:
        variant_indices_by_env = [0, 1]

        def get_bounding_box_per_env(self, num_envs):
            assert num_envs == 2
            return AxisAlignedBoundingBox(
                min_point=torch.tensor([[-0.1, -0.1, 0.0], [-0.2, -0.2, 0.0]]),
                max_point=torch.tensor([[0.1, 0.1, 0.2], [0.2, 0.2, 0.4]]),
            )

    geometry = object_geometry(
        _FakeEnv({"set": runtime_asset}, torch.zeros(2, 3), arena_scene_assets={"set": _ArenaAsset()}), "set"
    )

    bbox = geometry.aabb_w()
    assert torch.allclose(bbox.min_point, torch.tensor([[0.9, -0.1, 0.5], [1.8, -0.2, 0.5]]))
    assert torch.allclose(bbox.max_point, torch.tensor([[1.1, 0.1, 0.7], [2.2, 0.2, 0.9]]))


def test_object_geometry_falls_back_to_point_for_spawner_assets_without_bbox() -> None:
    from isaaclab_arena.scene.object_geometry import object_geometry

    root_pos_w = torch.tensor([[1.0, 2.0, 3.0]])
    runtime_asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=root_pos_w,
            root_lin_vel_w=torch.zeros(1, 3),
        )
    )

    def _unexpected_bbox():
        raise AssertionError("spawner-backed asset should not query USD bbox")

    arena_asset = types.SimpleNamespace(usd_path=None, get_bounding_box=_unexpected_bbox)
    geometry = object_geometry(
        _FakeEnv({"sphere": runtime_asset}, torch.zeros(1, 3), arena_scene_assets={"sphere": arena_asset}), "sphere"
    )

    bbox = geometry.aabb_w()
    assert torch.allclose(bbox.min_point, root_pos_w)
    assert torch.allclose(bbox.max_point, root_pos_w)


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


def test_geometry_destination_predicate_rejects_centroid_false_positive() -> None:
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.scene.object_geometry import object_geometry
    from isaaclab_arena.tasks.predicates.spatial import object_on_destination_by_geometry
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    soft_nodes = torch.tensor([[
        [0.0, 0.0, 0.11],
        [0.02, 0.0, 0.11],
        [0.04, 0.0, 0.11],
        [0.80, 0.0, 0.11],
    ]])
    env = _FakeEnv(
        {
            "soft": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    nodal_pos_w=soft_nodes,
                    nodal_vel_w=torch.zeros_like(soft_nodes),
                    root_pos_w=soft_nodes.mean(dim=1),
                    root_vel_w=torch.zeros(1, 3),
                )
            ),
            "tray": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pose_w=torch.tensor([[0.02, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]]),
                    root_pos_w=torch.tensor([[0.02, 0.0, 0.05]]),
                    root_lin_vel_w=torch.zeros(1, 3),
                )
            ),
        },
        torch.zeros(1, 3),
        arena_scene_assets={
            "tray": types.SimpleNamespace(
                get_bounding_box=lambda: AxisAlignedBoundingBox(
                    min_point=(-0.05, -0.05, -0.05), max_point=(0.05, 0.05, 0.05)
                )
            )
        },
    )

    assert torch.allclose(object_geometry(env, "soft").centroid_w(), torch.tensor([[0.215, 0.0, 0.11]]))
    assert not object_on_destination_by_geometry(
        env,
        object_cfg=SceneEntityCfg("soft"),
        target_object_cfg=SceneEntityCfg("tray"),
        support_tolerance=0.02,
        containment_margin=0.0,
        containment_fraction_threshold=1.0,
    ).item()
