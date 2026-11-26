# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChoiceSampler drawing from a fixed set, and for numeric variations taking a ChoiceSampler.

These tests stay in plain Python (no ``SimulationApp``): they draw from samplers and build cfgs without starting Isaac
Sim.
"""

import torch

import pytest

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.object_mass_variation import (
    ApplyObjectMassFromSampler,
    ObjectMassVariation,
    ObjectMassVariationCfg,
)
from isaaclab_arena.variations.sampler_base import sample_float_rows
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

MASS_VALUES = [[0.17], [0.24]]


def _seeded_sampler(cfg: ChoiceSamplerCfg, seed: int = 7) -> ChoiceSampler:
    sampler = cfg.build()
    sampler.set_generator(torch.Generator().manual_seed(seed))
    return sampler


def test_choice_sampler_draws_per_call_choices_unchanged_for_a_seed():
    # Literal values from ChoiceSampler before it took values and probabilities: recorded seeds must keep their draws.
    sampler = ChoiceSampler()
    sampler.set_generator(torch.Generator().manual_seed(7))

    assert sampler.sample(12, choices=list(range(5))) == [0, 2, 1, 1, 3, 2, 2, 4, 3, 1, 3, 1]
    assert sampler.sample(4, choices=["a", "b", "c"]) == ["c", "a", "c", "b"]


def test_choice_sampler_draws_fixed_values_for_a_seed():
    # Literal values: a seed that repeats only within one process would pass a comparison of two samplers.
    values = [[0.17], [0.24], [0.5]]
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=values))

    assert sampler.sample(8) == [[0.17], [0.24], [0.24], [0.5], [0.24], [0.17], [0.5], [0.5]]
    first_draws = _seeded_sampler(ChoiceSamplerCfg(values=values), seed=11).sample(16)
    assert _seeded_sampler(ChoiceSamplerCfg(values=values), seed=11).sample(16) == first_draws


def test_weighted_draws_do_not_depend_on_how_draws_are_split():
    cfg = ChoiceSamplerCfg(values=["a", "b", "c", "d"], probabilities=[0.1, 0.0, 0.6, 0.3])
    one_draw = _seeded_sampler(cfg, seed=3).sample(12)

    sampler = _seeded_sampler(cfg, seed=3)
    split_draws = [entry for num_samples in (1, 4, 7) for entry in sampler.sample(num_samples)]

    assert split_draws == one_draw


@pytest.mark.parametrize("random_choice", [True, False])
def test_numeric_choice_samples_match_uniform_sampler_rows(random_choice: bool):
    values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    choice_sampler = _seeded_sampler(ChoiceSamplerCfg(values=values, random_choice=random_choice))
    uniform_sampler = UniformSamplerCfg(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0]).build()

    choice_rows = sample_float_rows(choice_sampler, 5, (3,))
    uniform_rows = sample_float_rows(uniform_sampler, 5, (3,))

    assert choice_rows.shape == uniform_rows.shape == (5, 3)
    assert choice_rows.dtype == uniform_rows.dtype == torch.float32
    for row in choice_rows.tolist():
        assert row == pytest.approx(values[0]) or row == pytest.approx(values[1])
    assert sample_float_rows(choice_sampler, 0, (3,)).shape == (0, 3)


def test_sample_float_rows_rejects_values_of_the_wrong_shape():
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=[0.17, 0.24]))

    with pytest.raises(AssertionError, match=r"needs samples of shape \(1,\)"):
        sample_float_rows(sampler, 2, (1,))


def test_choice_sampler_samples_are_always_in_the_values():
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=MASS_VALUES, probabilities=[0.3, 0.7]))

    samples = sampler.sample(500)

    assert all(sample in MASS_VALUES for sample in samples)
    assert sampler.sample(0) == []


def test_choice_sampler_draws_from_its_generator_only():
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=MASS_VALUES, probabilities=[0.5, 0.5]))
    global_random_state = torch.get_rng_state()

    sampler.sample(16)

    assert torch.equal(torch.get_rng_state(), global_random_state)


def test_choice_sampler_follows_probabilities():
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=["a", "b", "c"], probabilities=[0.2, 0.0, 0.8]), seed=0)
    num_samples = 20000

    samples = sampler.sample(num_samples)

    assert "b" not in samples, "An entry with probability 0 must never be drawn."
    assert samples.count("a") / num_samples == pytest.approx(0.2, abs=0.02)
    assert samples.count("c") / num_samples == pytest.approx(0.8, abs=0.02)


def test_choice_sampler_without_probabilities_draws_entries_equally_often():
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=MASS_VALUES), seed=0)

    samples = sampler.sample(20000)

    assert samples.count([0.17]) / len(samples) == pytest.approx(0.5, abs=0.02)


def test_choice_sampler_per_env_mode_gives_env_i_entry_i_modulo_the_values():
    values = [[0.17], [0.24], [0.5]]
    sampler = ChoiceSamplerCfg(values=values, random_choice=False).build()
    generator = torch.Generator().manual_seed(7)
    sampler.set_generator(generator)
    generator_state = generator.get_state()

    all_envs = sampler.sample(5, env_ids=torch.arange(5, dtype=torch.int32))
    partial_reset = sampler.sample(2, env_ids=torch.tensor([4, 1]))
    without_env_ids = sampler.sample(4)

    assert all_envs == [[0.17], [0.24], [0.5], [0.17], [0.24]]
    assert partial_reset == [[0.24], [0.24]]
    assert without_env_ids == [[0.17], [0.24], [0.5], [0.17]]
    assert torch.equal(generator.get_state(), generator_state), "The per-env mode must not draw random numbers."


@pytest.mark.parametrize("random_choice", [True, False])
def test_choice_sampler_replays_recorded_values_and_notifies_listeners(random_choice: bool):
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=MASS_VALUES, random_choice=random_choice))
    received: list[tuple[list, torch.Tensor | None]] = []
    sampler.add_listener(lambda sample, env_ids: received.append((sample, env_ids)))
    env_ids = torch.tensor([0, 1])

    live = sampler.sample(2, env_ids=env_ids)
    sampler.set_replay_sampler(lambda _count, _env_ids: [[0.24], [0.24]])
    replayed = sampler.sample(2, env_ids=env_ids)

    assert replayed == [[0.24], [0.24]]
    assert len(received) == 2
    assert received[0][0] is live and received[0][1] is env_ids
    assert received[1][0] is replayed and received[1][1] is env_ids


def test_choice_sampler_falls_back_to_live_draws_when_replay_returns_none():
    sampler = _seeded_sampler(ChoiceSamplerCfg(values=MASS_VALUES, random_choice=False))
    sampler.set_replay_sampler(lambda _count, _env_ids: None)

    assert sampler.sample(2) == [[0.17], [0.24]]


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (ChoiceSamplerCfg(values=MASS_VALUES, probabilities=[1.0]), "one probability per value"),
        (ChoiceSamplerCfg(values=MASS_VALUES, probabilities=[1.5, -0.5]), "non-negative"),
        (ChoiceSamplerCfg(values=MASS_VALUES, probabilities=[0.5, 0.4]), "sum to 1"),
        (ChoiceSamplerCfg(values=MASS_VALUES, probabilities=[0.5, 0.5], random_choice=False), "random_choice"),
    ],
)
def test_choice_sampler_rejects_invalid_cfg(cfg: ChoiceSamplerCfg, message: str):
    with pytest.raises(AssertionError, match=message):
        cfg.build()


def test_choice_sampler_rejects_a_draw_without_entries_or_with_two_sources():
    with pytest.raises(AssertionError, match="non-empty"):
        ChoiceSamplerCfg().build().sample(1)
    with pytest.raises(AssertionError, match="has its own values"):
        ChoiceSamplerCfg(values=MASS_VALUES).build().sample(1, choices=["a"])
    with pytest.raises(AssertionError, match="one probability per entry"):
        ChoiceSamplerCfg(probabilities=[0.5, 0.5]).build().sample(1, choices=["a", "b", "c"])


def test_choice_sampler_cfg_drives_object_mass_variation():
    cfg = ObjectMassVariationCfg(sampler_cfg=ChoiceSamplerCfg(values=MASS_VALUES, random_choice=False))

    variation = ObjectMassVariation(asset_name="apple", cfg=cfg)
    variation.configure_at_build_time(seed=0, variation_key="apple.mass")
    _, event_cfg = variation.build_event_cfg()

    assert event_cfg.func is ApplyObjectMassFromSampler
    sampler = event_cfg.params["sampler"]
    assert isinstance(sampler, ChoiceSampler)
    # ApplyObjectMassFromSampler draws one scalar mass per env.
    torch.testing.assert_close(
        sample_float_rows(sampler, 3, (1,), torch.arange(3)), torch.tensor([[0.17], [0.24], [0.17]])
    )
