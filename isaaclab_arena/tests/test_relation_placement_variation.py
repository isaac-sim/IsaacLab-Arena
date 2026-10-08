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
    RelationPlacementVariation,
    RelationPlacementVariationCfg,
)


class _ReplayAsset:
    def __init__(self, *keys: str) -> None:
        self._keys = keys
        self.name = keys[0]

    def get_scene_root_keys(self) -> tuple[str, ...]:
        return self._keys

    def get_scene_key(self) -> str:
        return self._keys[0]

    def get_relations(self) -> list:
        return []

    def has_pose_reset_event(self) -> bool:
        return False


def _placement_sample(*, asset: str = "cube_asset") -> dict:
    return {
        "layout_id": "layout_000000",
        "source": "test",
        "poses": {
            "cube": {
                "position_xyz": [0.1, 0.2, 0.3],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "assets": {"cube": asset},
    }


def _make_variation(identity: str = "cube_asset") -> RelationPlacementVariation:
    sampler = PlacementPoolSampler(
        assets=[],
        placement_pool=None,
        replay_assets=[_ReplayAsset("cube")],
        asset_identities={"cube": identity},
    )
    return RelationPlacementVariation(sampler, write_live_samples=True)


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


def test_replay_validates_scene_keys_and_concrete_asset_identity():
    variation = _make_variation()
    variation.validate_replay_samples([_placement_sample()])
    legacy = _placement_sample()
    del legacy["assets"]
    variation.validate_replay_samples([legacy])

    with pytest.raises(AssertionError, match="concrete assets differ"):
        variation.validate_replay_samples([_placement_sample(asset="different_asset")])
    unknown = _placement_sample()
    unknown["poses"]["unknown"] = unknown["poses"].pop("cube")
    unknown["assets"]["unknown"] = unknown["assets"].pop("cube")
    with pytest.raises(AssertionError, match="unknown=.*unknown"):
        variation.validate_replay_samples([unknown])


def test_replay_requires_every_root_of_a_selected_compound_asset():
    sampler = PlacementPoolSampler(
        assets=[],
        placement_pool=Mock(),
        replay_assets=[_ReplayAsset("left_robot", "right_robot")],
    )
    variation = RelationPlacementVariation(sampler, write_live_samples=True)
    sample = _placement_sample()
    sample["poses"] = {"left_robot": sample["poses"]["cube"]}
    sample.pop("assets")

    with pytest.raises(AssertionError, match="right_robot"):
        variation.validate_replay_samples([sample])


def test_relation_placement_cannot_be_disabled():
    variation = _make_variation()

    with pytest.raises(AssertionError, match="disable relation solving"):
        variation.apply_cfg(RelationPlacementVariationCfg(enabled=False))


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
