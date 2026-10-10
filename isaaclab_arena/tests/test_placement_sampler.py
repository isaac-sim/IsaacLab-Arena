# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for sampler-level relation placement replay."""

import torch
from unittest.mock import Mock

import pytest

from isaaclab_arena.relations.placement_sampler import (
    PlacementSample,
    PlacementSampler,
    deserialize_placement_samples,
    serialize_placement_samples,
    supports_recorded_placement,
)


class _ReplayAsset:
    def __init__(self, *keys: str) -> None:
        self._keys = keys
        self.name = keys[0]

    def get_scene_root_keys(self) -> tuple[str, ...]:
        return self._keys


def _record() -> dict:
    return {
        "layout_id": "layout_000000",
        "poses": {
            "cube": {
                "position_xyz": [0.1, 0.2, 0.3],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
    }


def test_replay_sampler_returns_native_samples_and_notifies_listener() -> None:
    sampler = PlacementSampler(assets=[], placement_pool=None, write_assets=[_ReplayAsset("cube")])
    recorded_sample = PlacementSample.from_record(_record())
    sampler.set_replay_sampler(lambda count, env_ids: [recorded_sample] * count)
    observed = []
    sampler.add_listener(lambda samples, env_ids: observed.append((samples, env_ids.clone())))

    samples = sampler.sample(2, torch.tensor([2, 5]))

    assert all(isinstance(sample, PlacementSample) for sample in samples)
    assert [sample.layout_id for sample in samples] == ["layout_000000", "layout_000000"]
    assert serialize_placement_samples(samples) == [_record(), _record()]
    assert observed[0][0] is samples
    assert torch.equal(observed[0][1], torch.tensor([2, 5]))


def test_fixed_live_sampler_reuses_per_environment_samples() -> None:
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.validation.types import PlacementValidationResults

    asset = Mock()
    asset.get_relations.return_value = []
    asset.get_scene_root_keys.return_value = ("cube",)
    asset.layout_pose_to_scene_writes.side_effect = lambda pose: [("cube", pose)]
    layouts = [
        PlacementResult(
            validation_results=PlacementValidationResults(),
            positions={asset: (float(env_id), 0.0, 0.2)},
            final_loss=0.0,
            attempts=1,
        )
        for env_id in range(2)
    ]
    pool = Mock()
    pool.sample_with_replacement.return_value = layouts
    sampler = PlacementSampler(assets=[asset], placement_pool=pool)
    sampler.prepare_live(num_envs=2, resolve_on_reset=False)

    first = sampler.sample(1, torch.tensor([1]))
    second = sampler.sample(1, torch.tensor([1]))

    pool.sample_for_envs.assert_not_called()
    assert first == second
    assert first[0].poses["cube"].position_xyz == (1.0, 0.0, 0.2)


def test_deserialize_placement_samples_rejects_inconsistent_roots() -> None:
    second = _record()
    second["poses"] = {"other": second["poses"]["cube"]}

    with pytest.raises(AssertionError, match="same scene roots"):
        deserialize_placement_samples([_record(), second])


def test_recorded_placement_does_not_support_rigid_object_sets() -> None:
    from isaaclab_arena.assets.object_set import RigidObjectSet

    object_set = RigidObjectSet.__new__(RigidObjectSet)

    assert supports_recorded_placement([Mock()])
    assert not supports_recorded_placement([Mock(), object_set])
