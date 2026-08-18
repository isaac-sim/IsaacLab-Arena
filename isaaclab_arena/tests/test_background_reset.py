# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Replicator background reset state."""

import torch

import warp as wp
from pxr import Sdf

from isaaclab_arena.terms.background_reset import (
    _ArticulationReset,
    _exclude_articulation_owned_bodies,
    _RigidBodyReset,
)


class _RecordingRigidBodyView:
    def set_transforms(self, values, indices) -> None:
        self.transforms = wp.to_torch(values).clone()
        self.transform_indices = wp.to_torch(indices).clone()

    def set_velocities(self, values, indices) -> None:
        self.velocities = wp.to_torch(values).clone()
        self.velocity_indices = wp.to_torch(indices).clone()


class _RecordingArticulationView:
    def __init__(self) -> None:
        self.calls = {}

    def _record(self, name, values, indices) -> None:
        self.calls[name] = (wp.to_torch(values).clone(), wp.to_torch(indices).clone())

    def set_root_transforms(self, values, indices) -> None:
        self._record("root_transforms", values, indices)

    def set_root_velocities(self, values, indices) -> None:
        self._record("root_velocities", values, indices)

    def set_dof_positions(self, values, indices) -> None:
        self._record("dof_positions", values, indices)

    def set_dof_velocities(self, values, indices) -> None:
        self._record("dof_velocities", values, indices)


def test_replicator_body_event_is_independent_of_root_pose_reset():
    from isaaclab_arena.assets.background_library import ReplicatorKitchenLShape
    from isaaclab_arena.assets.object_base import ObjectType

    background = ReplicatorKitchenLShape.__new__(ReplicatorKitchenLShape)
    background.object_type = ObjectType.BASE
    background.reset_pose = True
    background._pose_event_cfg = None

    assert not background.has_pose_reset_event()
    event_name, event_cfg = background.get_event_cfg()
    assert event_name == "replicator_kitchen_l_shape_body_reset"
    assert event_cfg.params["background_name"] == background.name

    background.disable_reset_pose()
    assert not background.has_pose_reset_event()
    assert background.get_event_cfg()[0] == event_name


def test_exclude_articulation_owned_bodies_partitions_by_subtree():
    body_paths = (
        "/World/envs/env_0/kitchen/Range/body",
        "/World/envs/env_0/kitchen/Fridge/door",
    )
    articulation_roots = (Sdf.Path("/World/envs/env_0/kitchen/Range"),)

    assert _exclude_articulation_owned_bodies(body_paths, articulation_roots) == (
        "/World/envs/env_0/kitchen/Fridge/door",
    )


def test_rigid_body_reset_uses_full_buffers_with_selected_indices():
    view = _RecordingRigidBodyView()
    transforms = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    zero_velocities = torch.zeros((2, 6))
    indices = wp.from_torch(torch.tensor([1], dtype=torch.int32))

    _RigidBodyReset(view=view, transforms=transforms).restore(indices, zero_velocities)

    assert torch.equal(view.transforms, transforms)
    assert torch.equal(view.velocities, zero_velocities)
    assert view.transform_indices.tolist() == [1]
    assert view.velocity_indices.tolist() == [1]


def test_articulation_reset_restores_pose_and_clears_velocities():
    view = _RecordingArticulationView()
    root_transforms = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    dof_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    zero_root_velocities = torch.zeros((2, 6))
    indices = wp.from_torch(torch.tensor([0], dtype=torch.int32))

    _ArticulationReset(view=view, root_transforms=root_transforms, dof_positions=dof_positions).restore(
        indices, zero_root_velocities
    )

    assert torch.equal(view.calls["root_transforms"][0], root_transforms)
    assert torch.equal(view.calls["root_velocities"][0], zero_root_velocities)
    assert torch.equal(view.calls["dof_positions"][0], dof_positions)
    assert torch.equal(view.calls["dof_velocities"][0], torch.zeros_like(dof_positions))
    assert all(call_indices.tolist() == [0] for _, call_indices in view.calls.values())
