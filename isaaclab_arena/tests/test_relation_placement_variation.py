# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for scene-level relation placement variation samples."""

import json
import torch
from unittest.mock import Mock

import pytest

from isaaclab_arena.variations.object_mass_variation import ObjectMassVariation, ObjectMassVariationCfg
from isaaclab_arena.variations.recorded_variation_replay import configure_recorded_variation_replay
from isaaclab_arena.variations.relation_placement_variation import (
    PlacementPoolSampler,
    RelationPlacementHandle,
    RelationPlacementVariation,
    RelationPlacementVariationCfg,
    apply_relation_placement_sample,
)


class _ReplayAsset:
    def __init__(self, *keys: str) -> None:
        self._keys = keys
        self.name = keys[0]
        self.scene_pose_writes = []
        self.initial_scene_root_poses = None

    def get_scene_root_keys(self) -> tuple[str, ...]:
        return self._keys

    def get_scene_key(self) -> str:
        return self._keys[0]

    def get_relations(self) -> list:
        return []

    def layout_pose_to_scene_writes(self, layout_pose):
        return [(self.get_scene_key(), layout_pose)]

    def has_pose_reset_event(self) -> bool:
        return False

    def write_scene_root_poses_to_sim(self, env, env_ids, poses) -> None:
        self.scene_pose_writes.append((env_ids.clone(), poses))

    def set_initial_scene_root_poses(self, poses) -> None:
        self.initial_scene_root_poses = poses


def _placement_sample() -> dict:
    return {
        "layout_id": "layout_000000",
        "source": "test",
        "poses": {
            "cube": {
                "position_xyz": [0.1, 0.2, 0.3],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
    }


def _make_variation() -> RelationPlacementVariation:
    sampler = PlacementPoolSampler(
        assets=[],
        placement_pool=None,
        replay_assets=[_ReplayAsset("cube")],
    )
    return RelationPlacementVariation(sampler, num_envs=1)


def test_replay_sampler_notifies_serializable_rows():
    variation = _make_variation()
    assert not variation.has_live_pool
    sample = _placement_sample()
    observed = []
    variation.add_sample_listener(lambda rows, env_ids: observed.append((rows, env_ids.clone())))
    variation.set_replay_sampler(lambda count, env_ids: [sample] * count)

    env_ids = torch.tensor([2, 5])
    rows = variation.sampler.sample(2, env_ids)

    assert rows == [sample, sample]
    assert observed[0][0] == rows
    assert observed[0][1].tolist() == [2, 5]
    assert variation.last_results == {}


def test_replay_writes_poses_through_placement_asset():
    asset = _ReplayAsset("cube")
    sampler = PlacementPoolSampler(assets=[], placement_pool=None, replay_assets=[asset])
    sampler.set_replay_sampler(lambda count, env_ids: [_placement_sample()] * count)
    placement = RelationPlacementHandle(sampler)
    env = Mock(device=torch.device("cpu"))
    env_ids = torch.tensor([1, 3])

    apply_relation_placement_sample(env, env_ids, placement)

    written_env_ids, written_poses = asset.scene_pose_writes[0]
    assert written_env_ids.tolist() == [1, 3]
    torch.testing.assert_close(
        written_poses["cube"],
        torch.tensor([
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        ]),
    )


def test_fixed_live_placement_writes_through_variation_without_sampling_pool():
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.validation.types import PlacementValidationResults

    asset = _ReplayAsset("cube")
    results = {
        env_id: PlacementResult(
            validation_results=PlacementValidationResults(),
            positions={asset: (float(env_id), 0.0, 0.2)},
            final_loss=0.0,
            attempts=1,
        )
        for env_id in range(2)
    }
    placement_pool = Mock(num_envs=2)
    sampler = PlacementPoolSampler(
        assets=[asset],
        placement_pool=placement_pool,
    )
    placement_pool.sample_with_replacement.return_value = list(results.values())
    sampler.prepare_live(num_envs=2, resample_on_reset=False)
    env = Mock(device=torch.device("cpu"))
    env.scene.env_origins = torch.zeros((2, 3))

    apply_relation_placement_sample(env, torch.tensor([1]), RelationPlacementHandle(sampler))

    placement_pool.sample_for_envs.assert_not_called()
    written_env_ids, written_poses = asset.scene_pose_writes[0]
    assert written_env_ids.tolist() == [1]
    torch.testing.assert_close(
        written_poses["cube"],
        torch.tensor([[1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]),
    )


def test_replay_validates_scene_keys():
    variation = _make_variation()
    variation.validate_replay_samples([_placement_sample()])
    unknown = _placement_sample()
    unknown["poses"]["unknown"] = unknown["poses"].pop("cube")
    with pytest.raises(AssertionError, match="unknown=.*unknown"):
        variation.validate_replay_samples([unknown])


def test_replay_requires_every_root_of_a_selected_compound_asset():
    sampler = PlacementPoolSampler(
        assets=[],
        placement_pool=Mock(),
        replay_assets=[_ReplayAsset("left_robot", "right_robot")],
    )
    variation = RelationPlacementVariation(sampler, num_envs=1)
    sample = _placement_sample()
    sample["poses"] = {"left_robot": sample["poses"]["cube"]}

    with pytest.raises(AssertionError, match="right_robot"):
        variation.validate_replay_samples([sample])


def test_relation_placement_cannot_be_disabled():
    variation = _make_variation()

    with pytest.raises(AssertionError, match="disable relation solving"):
        variation.apply_cfg(RelationPlacementVariationCfg(enabled=False))


def test_replay_only_declaration_reports_sample_source_availability_without_disabling_itself():
    sampler = PlacementPoolSampler(
        assets=[],
        placement_pool=None,
        replay_assets=[_ReplayAsset("cube")],
    )
    variation = RelationPlacementVariation(
        sampler,
        num_envs=1,
    )

    variation.set_replay_sampler(None)

    assert variation.enabled
    assert not variation.can_supply_samples

    variation.set_replay_sampler(lambda count, env_ids: [_placement_sample()] * count)
    variation.on_replay_samples_bound([_placement_sample()])

    assert variation.enabled
    assert variation.can_supply_samples


def test_validated_replay_rows_are_available_during_preparation():
    asset = _ReplayAsset("cube")
    sampler = PlacementPoolSampler(
        assets=[],
        placement_pool=None,
        replay_assets=[asset],
    )
    variation = RelationPlacementVariation(
        sampler,
        num_envs=2,
    )
    sample = _placement_sample()
    variation.validate_replay_samples([sample])
    variation.set_replay_sampler(lambda count, env_ids: [sample] * count)
    variation.on_replay_samples_bound([sample])

    variation.configure_at_build_time()

    assert asset.initial_scene_root_poses is not None
    assert [pose.position_xyz for pose in asset.initial_scene_root_poses["cube"].poses] == [(0.1, 0.2, 0.3)] * 2


def test_placement_replays_with_another_runtime_condition(tmp_path):
    placement = _make_variation()
    mass = ObjectMassVariation("cube", ObjectMassVariationCfg(enabled=True))
    path = tmp_path / "variation_samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "variations": {
                    "scene.relation_placement": _placement_sample(),
                    "cube.mass": [0.75],
                }
            }
        )
        + "\n"
    )

    scheduler = configure_recorded_variation_replay(path, {"scene": [placement], "cube": [mass]})
    scheduler.assign_new_episodes([3])
    env_ids = torch.tensor([3])

    assert placement.sampler.sample(1, env_ids) == [_placement_sample()]
    torch.testing.assert_close(mass.sampler.sample(1, env_ids), torch.tensor([[0.75]]))
