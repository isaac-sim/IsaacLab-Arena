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

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation.episode_conditions_rollout import resolve_policy_runner_replay_budget
from isaaclab_arena.variations.bernoulli_sampler import BernoulliSampler
from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.condition_replay import (
    bind_condition_replay_samplers,
    validate_condition_replay_variations,
)
from isaaclab_arena.variations.condition_scheduler import ConditionScheduler
from isaaclab_arena.variations.recorded_variation_samples import (
    EpisodeVariationRecord,
    RebuildVariationRecord,
    load_rebuild_variation_record,
    validate_recorded_variation_sample_keys,
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


def test_condition_scheduler_cycles_globally_across_partial_resets() -> None:
    variation_record = _runtime_record([0, 1, 2])
    scheduler = ConditionScheduler(variation_record)
    observed_source_indices: list[int] = []

    scheduler.assign_new_episodes([0, 1])
    observed_source_indices.extend([
        scheduler.source_record_index_for_env(0),
        scheduler.source_record_index_for_env(1),
    ])
    for env_id in [1, 0, 0, 1, 0, 1]:
        scheduler.complete_episodes([env_id])
        scheduler.assign_new_episodes([env_id])
        observed_source_indices.append(scheduler.source_record_index_for_env(env_id))

    assert observed_source_indices == [0, 1, 2, 0, 1, 2, 0, 1]
    assert scheduler.num_assignments_started == 8


def test_repeated_initial_assignment_does_not_consume_condition() -> None:
    scheduler = ConditionScheduler(_runtime_record([0, 1, 2]))

    scheduler.assign_new_episodes([0, 1])
    scheduler.assign_new_episodes([0, 1])

    assert scheduler.num_assignments_started == 2
    assert scheduler.source_record_index_for_env(0) == 0
    assert scheduler.source_record_index_for_env(1) == 1


def test_condition_replay_replays_present_variation_and_samples_absent_live() -> None:
    replayed = _RunTimeTestVariation("replayed", live_value=9.0)
    live = _RunTimeTestVariation("live", live_value=7.0)
    variation_record = RebuildVariationRecord(
        build_time_samples={},
        episode_records=[
            EpisodeVariationRecord(runtime_samples={"asset.replayed": [0.1]}),
            EpisodeVariationRecord(runtime_samples={"asset.replayed": [0.2]}),
        ],
    )
    scheduler = ConditionScheduler(variation_record)
    scheduler.assign_new_episodes([0, 1])

    bind_condition_replay_samplers(
        {"asset": [replayed, live]},
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


def test_condition_replay_rejects_mixed_runtime_presence() -> None:
    variation = _RunTimeTestVariation("offset", live_value=0.0)
    variation_record = RebuildVariationRecord(
        build_time_samples={},
        episode_records=[
            EpisodeVariationRecord(runtime_samples={"asset.offset": [0.1]}),
            EpisodeVariationRecord(runtime_samples={}),
        ],
    )

    with pytest.raises(AssertionError, match="present in every source record or none"):
        validate_condition_replay_variations({"asset": [variation]}, variation_record)


def test_condition_replay_rejects_wrong_variation_lifecycle() -> None:
    runtime = _RunTimeTestVariation("runtime", live_value=0.0)
    build_time = _BuildTimeTestVariation("build_time", live_value=0.0)
    variation_record = RebuildVariationRecord(
        build_time_samples={"asset.runtime": [0.1]},
        episode_records=[
            EpisodeVariationRecord(runtime_samples={"asset.build_time": [0.2]}),
        ],
    )

    with pytest.raises(AssertionError, match="cannot appear"):
        validate_condition_replay_variations({"asset": [runtime, build_time]}, variation_record)


def test_policy_runner_replay_budget_defaults_to_exact_pass(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {"variations": {}},
            {"variations": {}},
            {"variations": {}},
        ],
    )
    cfg = ArenaEnvBuilderCfg(episode_conditions_path=str(path))

    assert resolve_policy_runner_replay_budget(cfg, num_steps=None, num_episodes=None) == 3


def test_policy_runner_replay_budget_honors_explicit_episode_count(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [{"variations": {}}, {"variations": {}}])
    cfg = ArenaEnvBuilderCfg(episode_conditions_path=str(path))

    assert resolve_policy_runner_replay_budget(cfg, num_steps=None, num_episodes=8) == 8
    with pytest.raises(AssertionError, match="does not support --num_steps"):
        resolve_policy_runner_replay_budget(cfg, num_steps=8, num_episodes=None)


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
        episode_records=[
            EpisodeVariationRecord(runtime_samples={"asset.value": value}) for value in values
        ],
    )


def _write_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path
