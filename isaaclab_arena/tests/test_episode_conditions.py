# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for episode-condition extraction and replayable sampler primitives."""

import json
import torch
from pathlib import Path
from typing import Any

import pytest

from isaaclab_arena.variations.bernoulli_sampler import BernoulliSampler
from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.condition_replay import (
    _bind_condition_replay_samplers,
    _enabled_variations_by_key,
    _validate_condition_replay_variations,
)
from isaaclab_arena.variations.condition_scheduler import ConditionScheduler
from isaaclab_arena.variations.episode_conditions import (
    EpisodeCondition,
    RebuildConditions,
    load_episode_conditions_overlay,
    load_runtime_variation_samples,
    validate_overlay_variation_keys,
)
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase, VariationBaseCfg


def test_loader_splits_build_time_and_runtime(tmp_path: Path) -> None:
    jsonl_path = _write_jsonl(
        tmp_path,
        [
            {"variations": {"light.hdr_image": "home_office", "pick_object.mass": [0.5]}},
            {"variations": {"light.hdr_image": "home_office", "pick_object.mass": [0.8]}},
        ],
    )

    overlay = load_episode_conditions_overlay(
        jsonl_path,
        build_time_variation_keys={"light.hdr_image"},
    )

    assert overlay.build_time_variations == {"light.hdr_image": "home_office"}
    assert [episode.runtime_variations for episode in overlay.episodes] == [
        {"pick_object.mass": [0.5]},
        {"pick_object.mass": [0.8]},
    ]
    assert overlay.num_conditions == 2
    validate_overlay_variation_keys(overlay, {"light.hdr_image", "pick_object.mass"})


@pytest.mark.parametrize("num_records", [1, 2])
def test_runtime_values_are_not_inferred_as_build_time(tmp_path: Path, num_records: int) -> None:
    jsonl_path = _write_jsonl(
        tmp_path,
        [{"variations": {"obj.mass": [1.0]}}] * num_records,
    )

    loaded = load_episode_conditions_overlay(jsonl_path)

    assert loaded.build_time_variations == {}
    assert [episode.runtime_variations for episode in loaded.episodes] == [{"obj.mass": [1.0]}] * num_records


def test_load_runtime_variation_samples_distinguishes_replay_from_live(tmp_path: Path) -> None:
    recorded = _write_jsonl(
        tmp_path,
        [
            {"variations": {"scene.relation_placement": {"layout_id": "0"}}},
            {"variations": {"scene.relation_placement": {"layout_id": "1"}}},
        ],
    )
    assert load_runtime_variation_samples(recorded, "scene.relation_placement") == [
        {"layout_id": "0"},
        {"layout_id": "1"},
    ]

    live = _write_jsonl(tmp_path, [{"variations": {"obj.mass": [1.0]}}])
    assert load_runtime_variation_samples(live, "scene.relation_placement") is None

    partial = _write_jsonl(
        tmp_path,
        [
            {"variations": {"scene.relation_placement": {"layout_id": "0"}}},
            {"variations": {}},
        ],
    )
    with pytest.raises(AssertionError, match="every source condition or none"):
        load_runtime_variation_samples(partial, "scene.relation_placement")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {"variations": {"light.hdr_image": "studio"}},
                {"variations": {}},
            ],
            "missing from some",
        ),
        (
            [
                {"variations": {"light.hdr_image": "studio"}},
                {"variations": {"light.hdr_image": "kitchen"}},
            ],
            "changes across",
        ),
    ],
)
def test_loader_rejects_inconsistent_build_time_values(
    tmp_path: Path,
    records: list[dict[str, Any]],
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        load_episode_conditions_overlay(
            _write_jsonl(tmp_path, records),
            build_time_variation_keys={"light.hdr_image"},
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.jsonl"
    path.write_text('{"variations":{"obj.mass":[1.0],"obj.mass":[2.0]}}\n')

    with pytest.raises(AssertionError, match="Duplicate key"):
        load_episode_conditions_overlay(path)


def test_loader_rejects_non_jsonl_input(tmp_path: Path) -> None:
    path = tmp_path / "conditions.yaml"
    path.write_text("episodes: []\n")

    with pytest.raises(AssertionError, match="must be loaded from JSONL"):
        load_episode_conditions_overlay(path)


def test_null_jsonl_variations_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "null.jsonl"
    path.write_text('{"variations": null}\n')

    with pytest.raises(AssertionError, match="variations must be a mapping"):
        load_episode_conditions_overlay(path)


def test_unknown_variation_keys_are_rejected(tmp_path: Path) -> None:
    overlay = load_episode_conditions_overlay(_write_jsonl(tmp_path, [{"variations": {"unknown.variation": [1.0]}}]))

    with pytest.raises(AssertionError, match="no enabled variation"):
        validate_overlay_variation_keys(overlay, set())


def test_variation_cannot_be_both_build_time_and_runtime() -> None:
    overlay = RebuildConditions(
        build_time_variations={"light.hdr_image": "studio"},
        episodes=[
            EpisodeCondition(
                condition_id="condition_000000",
                runtime_variations={"light.hdr_image": "kitchen"},
            )
        ],
    )

    with pytest.raises(AssertionError, match="both build-time and run-time"):
        validate_overlay_variation_keys(overlay, {"light.hdr_image"})


def test_replay_samplers_preserve_output_types() -> None:
    continuous = UniformSamplerCfg(low=[0.0], high=[0.0]).build()
    continuous.set_replay_sampler(lambda _count, _env_ids: [[0.25], [0.75]])
    torch.testing.assert_close(continuous.sample(2), torch.tensor([[0.25], [0.75]]))

    choice = ChoiceSampler()
    choice.set_replay_sampler(lambda _count, _env_ids: ["recorded"])
    assert choice.sample(1, choices=["live", "recorded"]) == ["recorded"]

    bernoulli = BernoulliSampler(probability=0.0)
    bernoulli.set_replay_sampler(lambda _count, _env_ids: [True, False])
    assert bernoulli.sample(2) == [True, False]


def test_replay_samplers_preserve_public_preconditions() -> None:
    continuous = UniformSamplerCfg(low=[0.0], high=[0.0]).build()
    continuous.set_replay_sampler(lambda _count, _env_ids: [])
    with pytest.raises(AssertionError, match="non-negative"):
        continuous.sample(-1)

    choice = ChoiceSampler()
    choice.set_replay_sampler(lambda _count, _env_ids: ["recorded"])
    with pytest.raises(AssertionError, match="non-empty"):
        choice.sample(1, choices=[])
    with pytest.raises(AssertionError, match="must belong"):
        choice.sample(1, choices=["live"])


def test_condition_scheduler_cycles_globally_across_partial_resets() -> None:
    overlay = _runtime_overlay([0, 1, 2])
    scheduler = ConditionScheduler(overlay)
    observed_values: list[int] = []

    scheduler.assign_new_episodes([0, 1])
    scheduler.assign_new_episodes([0, 1])
    assert scheduler.num_assignments_started == 2
    observed_values.extend([
        scheduler.condition_for_env(0).runtime_variations["asset.value"],
        scheduler.condition_for_env(1).runtime_variations["asset.value"],
    ])
    for env_id in [1, 0, 0, 1, 0, 1]:
        scheduler.complete_episodes([env_id])
        scheduler.assign_new_episodes([env_id])
        observed_values.append(scheduler.condition_for_env(env_id).runtime_variations["asset.value"])

    assert observed_values == [0, 1, 2, 0, 1, 2, 0, 1]
    assert scheduler.num_assignments_started == 8


def test_condition_replay_replays_present_variation_and_samples_absent_live() -> None:
    replayed = _RunTimeTestVariation("replayed", live_value=9.0)
    live = _RunTimeTestVariation("live", live_value=7.0)
    overlay = RebuildConditions(
        build_time_variations={},
        episodes=[
            EpisodeCondition("condition_0", {"asset.replayed": [0.1]}),
            EpisodeCondition("condition_1", {"asset.replayed": [0.2]}),
        ],
    )
    scheduler = ConditionScheduler(overlay)
    scheduler.assign_new_episodes([0, 1])

    _bind_condition_replay_samplers(
        _enabled_variations_by_key({"asset": [replayed, live]}),
        overlay,
        scheduler,
    )

    torch.testing.assert_close(
        replayed.sampler.sample(2, env_ids=torch.tensor([1, 0])),
        torch.tensor([[0.2], [0.1]]),
    )
    torch.testing.assert_close(
        live.sampler.sample(2, env_ids=torch.tensor([1, 0])),
        torch.tensor([[7.0], [7.0]]),
    )


def test_condition_replay_rejects_mixed_runtime_presence() -> None:
    variation = _RunTimeTestVariation("offset", live_value=0.0)
    overlay = RebuildConditions(
        build_time_variations={},
        episodes=[
            EpisodeCondition("condition_0", {"asset.offset": [0.1]}),
            EpisodeCondition("condition_1", {}),
        ],
    )

    with pytest.raises(AssertionError, match="present in every source condition or none"):
        _validate_condition_replay_variations(_enabled_variations_by_key({"asset": [variation]}), overlay)


@pytest.mark.parametrize("recorded_at_runtime", [False, True])
def test_condition_replay_rejects_wrong_variation_lifecycle(recorded_at_runtime: bool) -> None:
    variation: BuildTimeVariationBase | RunTimeVariationBase
    if recorded_at_runtime:
        variation = _BuildTimeTestVariation("value", live_value=0.0)
        overlay = RebuildConditions(
            build_time_variations={},
            episodes=[EpisodeCondition("condition_0", {"asset.value": [0.2]})],
        )
    else:
        variation = _RunTimeTestVariation("value", live_value=0.0)
        overlay = RebuildConditions(
            build_time_variations={"asset.value": [0.1]},
            episodes=[EpisodeCondition("condition_0", {})],
        )

    with pytest.raises(AssertionError, match="cannot appear"):
        _validate_condition_replay_variations(
            _enabled_variations_by_key({"asset": [variation]}),
            overlay,
        )


class _RunTimeTestVariation(RunTimeVariationBase):
    def __init__(self, name: str, live_value: float):
        super().__init__(
            VariationBaseCfg(
                enabled=True,
                sampler_cfg=UniformSamplerCfg(low=[live_value], high=[live_value]),
            ),
            name,
        )

    def build_event_cfg(self):
        raise NotImplementedError


class _BuildTimeTestVariation(BuildTimeVariationBase):
    def __init__(self, name: str, live_value: float):
        super().__init__(
            VariationBaseCfg(
                enabled=True,
                sampler_cfg=UniformSamplerCfg(low=[live_value], high=[live_value]),
            ),
            name,
        )

    def _realize_at_build_time(self) -> None:
        pass


def _runtime_overlay(values: list[int]) -> RebuildConditions:
    return RebuildConditions(
        build_time_variations={},
        episodes=[EpisodeCondition(f"condition_{index}", {"asset.value": value}) for index, value in enumerate(values)],
    )


def _write_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path
