# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recorded variation sample extraction and replayable sampler primitives."""

import json
import torch
from pathlib import Path
from typing import Any

import pytest

from isaaclab_arena.variations.bernoulli_sampler import BernoulliSampler
from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.recorded_variation_replay import (
    _bind_variation_replay_samplers,
    _enabled_variations_by_key,
    _validate_variation_replay,
)
from isaaclab_arena.variations.recorded_variation_samples import (
    EpisodeVariationRecord,
    RebuildVariationRecord,
    load_rebuild_variation_record,
    validate_recorded_variation_sample_keys,
)
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase, VariationBaseCfg
from isaaclab_arena.variations.variation_replay_scheduler import VariationReplayScheduler


def test_loader_splits_build_time_and_runtime(tmp_path: Path) -> None:
    jsonl_path = _write_jsonl(
        tmp_path,
        [
            {"variations": {"light.hdr_image": "home_office", "pick_object.mass": [0.5]}},
            {"variations": {"light.hdr_image": "home_office", "pick_object.mass": [0.8]}},
        ],
    )

    samples = load_rebuild_variation_record(
        jsonl_path,
        build_time_variation_keys={"light.hdr_image"},
    )

    assert samples.build_time_samples == {"light.hdr_image": "home_office"}
    assert [record.runtime_samples for record in samples.episode_records] == [
        {"pick_object.mass": [0.5]},
        {"pick_object.mass": [0.8]},
    ]
    assert samples.num_recorded_episodes == 2
    validate_recorded_variation_sample_keys(samples, {"light.hdr_image", "pick_object.mass"})


def test_loader_keeps_placement_outside_variations(tmp_path: Path) -> None:
    placements = [{"layout_id": f"layout_{index}", "source": "test", "poses": {"cube": {}}} for index in range(2)]
    loaded = load_rebuild_variation_record(
        _write_jsonl(
            tmp_path,
            [
                {"placement": placement, "variations": {"obj.mass": [index]}}
                for index, placement in enumerate(placements)
            ],
        ),
        build_time_variation_keys=set(),
    )

    assert loaded.has_placement_samples
    assert [record.placement_sample for record in loaded.episode_records] == placements
    assert [record.runtime_samples for record in loaded.episode_records] == [
        {"obj.mass": [0]},
        {"obj.mass": [1]},
    ]


def test_loader_rejects_placement_missing_from_some_rows(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [{"placement": {}}, {"variations": {}}])

    with pytest.raises(AssertionError, match="present in every"):
        load_rebuild_variation_record(path, build_time_variation_keys=set())


@pytest.mark.parametrize("num_records", [1, 2])
def test_runtime_values_are_not_inferred_as_build_time(tmp_path: Path, num_records: int) -> None:
    loaded = load_rebuild_variation_record(
        _write_jsonl(tmp_path, [{"variations": {"obj.mass": [1.0]}}] * num_records),
        build_time_variation_keys=set(),
    )

    assert loaded.build_time_samples == {}
    assert [record.runtime_samples for record in loaded.episode_records] == [{"obj.mass": [1.0]}] * num_records


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
        load_rebuild_variation_record(
            _write_jsonl(tmp_path, records),
            build_time_variation_keys={"light.hdr_image"},
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.jsonl"
    path.write_text('{"variations":{"obj.mass":[1.0],"obj.mass":[2.0]}}\n')

    with pytest.raises(AssertionError, match="Duplicate key"):
        load_rebuild_variation_record(path, build_time_variation_keys=set())


def test_loader_rejects_non_jsonl_input(tmp_path: Path) -> None:
    path = tmp_path / "variation_samples.yaml"
    path.write_text("episodes: []\n")

    with pytest.raises(AssertionError, match="must be loaded from JSONL"):
        load_rebuild_variation_record(path, build_time_variation_keys=set())


def test_null_jsonl_variations_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "null.jsonl"
    path.write_text('{"variations": null}\n')

    with pytest.raises(AssertionError, match="variations must be a mapping"):
        load_rebuild_variation_record(path, build_time_variation_keys=set())


def test_unknown_variation_keys_are_rejected(tmp_path: Path) -> None:
    samples = load_rebuild_variation_record(
        _write_jsonl(tmp_path, [{"variations": {"unknown.variation": [1.0]}}]),
        build_time_variation_keys=set(),
    )

    with pytest.raises(AssertionError, match="no enabled variation"):
        validate_recorded_variation_sample_keys(samples, set())


def test_variation_cannot_be_both_build_time_and_runtime() -> None:
    samples = RebuildVariationRecord(
        build_time_samples={"light.hdr_image": "studio"},
        episode_records=[
            EpisodeVariationRecord(
                runtime_samples={"light.hdr_image": "kitchen"},
            )
        ],
    )

    with pytest.raises(AssertionError, match="both build-time and run-time"):
        validate_recorded_variation_sample_keys(samples, {"light.hdr_image"})


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


def test_variation_replay_scheduler_cycles_globally_across_partial_resets() -> None:
    scheduler = VariationReplayScheduler(_runtime_record([0, 1, 2]))
    observed_values: list[int] = []

    scheduler.assign_new_episodes([0, 1])
    assert scheduler.num_assignments_started == 2
    observed_values.extend([
        scheduler.record_for_env(0).runtime_samples["asset.value"],
        scheduler.record_for_env(1).runtime_samples["asset.value"],
    ])
    for env_id in [1, 0, 0, 1, 0, 1]:
        scheduler.complete_episodes([env_id])
        scheduler.assign_new_episodes([env_id])
        observed_values.append(scheduler.record_for_env(env_id).runtime_samples["asset.value"])

    assert observed_values == [0, 1, 2, 0, 1, 2, 0, 1]
    assert scheduler.num_assignments_started == 8


def test_replay_scheduler_aligns_placement_with_variations() -> None:
    record = RebuildVariationRecord(
        build_time_samples={},
        episode_records=[
            EpisodeVariationRecord(
                runtime_samples={"obj.mass": [index]},
                placement_sample={"layout_id": f"layout_{index}"},
            )
            for index in range(3)
        ],
    )
    scheduler = VariationReplayScheduler(record)
    scheduler.assign_new_episodes([4, 1])

    assert scheduler.runtime_sample_for("obj.mass", [1, 4]) == [[1], [0]]
    assert scheduler.placement_sample_for([1, 4]) == [
        {"layout_id": "layout_1"},
        {"layout_id": "layout_0"},
    ]


def test_variation_replay_replays_present_variation_and_samples_absent_live() -> None:
    replayed = _RunTimeTestVariation("replayed", live_value=9.0)
    live = _RunTimeTestVariation("live", live_value=7.0)
    variation_record = RebuildVariationRecord(
        build_time_samples={},
        episode_records=[
            EpisodeVariationRecord(runtime_samples={"asset.replayed": [0.1]}),
            EpisodeVariationRecord(runtime_samples={"asset.replayed": [0.2]}),
        ],
    )
    scheduler = VariationReplayScheduler(variation_record)
    scheduler.assign_new_episodes([0, 1])

    _bind_variation_replay_samplers(
        _enabled_variations_by_key({"asset": [replayed, live]}),
        variation_record,
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


def test_variation_replay_rejects_mixed_runtime_presence() -> None:
    variation = _RunTimeTestVariation("offset", live_value=0.0)
    variation_record = RebuildVariationRecord(
        build_time_samples={},
        episode_records=[
            EpisodeVariationRecord(runtime_samples={"asset.offset": [0.1]}),
            EpisodeVariationRecord(runtime_samples={}),
        ],
    )

    with pytest.raises(AssertionError, match="present in every source record or none"):
        _validate_variation_replay(
            _enabled_variations_by_key({"asset": [variation]}),
            variation_record,
        )


@pytest.mark.parametrize("recorded_at_runtime", [False, True])
def test_variation_replay_rejects_wrong_variation_lifecycle(recorded_at_runtime: bool) -> None:
    variation: BuildTimeVariationBase | RunTimeVariationBase
    if recorded_at_runtime:
        variation = _BuildTimeTestVariation("value", live_value=0.0)
        variation_record = RebuildVariationRecord(
            build_time_samples={},
            episode_records=[EpisodeVariationRecord(runtime_samples={"asset.value": [0.2]})],
        )
    else:
        variation = _RunTimeTestVariation("value", live_value=0.0)
        variation_record = RebuildVariationRecord(
            build_time_samples={"asset.value": [0.1]},
            episode_records=[EpisodeVariationRecord(runtime_samples={})],
        )

    with pytest.raises(AssertionError, match="cannot appear"):
        _validate_variation_replay(
            _enabled_variations_by_key({"asset": [variation]}),
            variation_record,
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


def _runtime_record(values: list[int]) -> RebuildVariationRecord:
    return RebuildVariationRecord(
        build_time_samples={},
        episode_records=[EpisodeVariationRecord(runtime_samples={"asset.value": value}) for value in values],
    )


def _write_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path
