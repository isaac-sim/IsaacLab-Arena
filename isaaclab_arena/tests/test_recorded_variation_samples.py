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
from isaaclab_arena.variations.recorded_variation_samples import (
    EpisodeVariationRecord,
    RebuildVariationRecord,
    load_rebuild_variation_record,
    validate_recorded_variation_sample_keys,
)
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg


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


def test_constant_runtime_values_are_not_inferred_as_build_time(tmp_path: Path) -> None:
    jsonl_path = _write_jsonl(
        tmp_path,
        [
            {"variations": {"obj.mass": [1.0]}},
            {"variations": {"obj.mass": [1.0]}},
        ],
    )

    loaded = load_rebuild_variation_record(jsonl_path, build_time_variation_keys=set())

    assert loaded.build_time_samples == {}
    assert [record.runtime_samples for record in loaded.episode_records] == [
        {"obj.mass": [1.0]},
        {"obj.mass": [1.0]},
    ]


def test_single_row_runtime_value_is_not_inferred_as_build_time(tmp_path: Path) -> None:
    loaded = load_rebuild_variation_record(
        _write_jsonl(tmp_path, [{"variations": {"obj.mass": [1.0]}}]),
        build_time_variation_keys=set(),
    )

    assert loaded.build_time_samples == {}
    assert loaded.episode_records[0].runtime_samples == {"obj.mass": [1.0]}


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
    path = tmp_path / "conditions.yaml"
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


def _write_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path
