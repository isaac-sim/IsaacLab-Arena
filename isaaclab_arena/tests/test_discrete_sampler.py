# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DiscreteSampler and the index draw it shares with ChoiceSampler.

These tests stay in plain Python (no ``SimulationApp``): they draw from samplers and build cfgs without starting Isaac
Sim.
"""

import torch

import pytest

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, draw_random_indices
from isaaclab_arena.variations.discrete_sampler import DiscreteSampler, DiscreteSamplerCfg
from isaaclab_arena.variations.object_mass_variation import (
    ApplyObjectMassFromSampler,
    ObjectMassVariation,
    ObjectMassVariationCfg,
)
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

MASS_VALUES = [[0.17], [0.24]]


def _seeded_sampler(cfg: DiscreteSamplerCfg, seed: int = 7) -> DiscreteSampler:
    sampler = cfg.build()
    sampler.set_generator(torch.Generator().manual_seed(seed))
    return sampler


def _assert_rows_in_values(samples: torch.Tensor, values: list[list[float]]) -> None:
    values_tensor = torch.tensor(values, dtype=torch.float32)
    for row in samples:
        assert (row == values_tensor).all(dim=1).any(), f"Sampled row {row.tolist()} is not one of {values}."


def test_choice_sampler_draws_are_unchanged_by_the_shared_index_draw():
    # Literal values from ChoiceSampler before it shared draw_random_indices: recorded seeds must keep their draws.
    sampler = ChoiceSampler()
    sampler.set_generator(torch.Generator().manual_seed(7))

    assert sampler.sample(12, choices=list(range(5))) == [0, 2, 1, 1, 3, 2, 2, 4, 3, 1, 3, 1]
    assert sampler.sample(4, choices=["a", "b", "c"]) == ["c", "a", "c", "b"]


def test_weighted_index_draws_do_not_depend_on_how_draws_are_split():
    probabilities = torch.tensor([0.1, 0.0, 0.6, 0.3], dtype=torch.float64)
    one_draw = draw_random_indices(12, 4, torch.Generator().manual_seed(3), probabilities)

    generator = torch.Generator().manual_seed(3)
    split_draws = [draw_random_indices(num_samples, 4, generator, probabilities) for num_samples in (1, 4, 7)]

    assert torch.equal(torch.cat(split_draws), one_draw)


@pytest.mark.parametrize("random_choice", [True, False])
def test_discrete_sampler_returns_uniform_sampler_shape_and_dtype(random_choice: bool):
    values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    discrete_sampler = _seeded_sampler(DiscreteSamplerCfg(values=values, random_choice=random_choice))
    uniform_sampler = UniformSamplerCfg(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0]).build()

    discrete_samples = discrete_sampler.sample(5)
    uniform_samples = uniform_sampler.sample(5)

    assert discrete_sampler.shape_per_sample == uniform_sampler.shape_per_sample == torch.Size([3])
    assert discrete_samples.shape == uniform_samples.shape == (5, 3)
    assert discrete_samples.dtype == uniform_samples.dtype == torch.float32
    _assert_rows_in_values(discrete_samples, values)


def test_discrete_sampler_samples_are_always_in_the_values():
    sampler = _seeded_sampler(DiscreteSamplerCfg(values=MASS_VALUES, probabilities=[0.3, 0.7]))

    samples = sampler.sample(500)

    _assert_rows_in_values(samples, MASS_VALUES)
    assert sampler.sample(0).shape == (0, 1)


def test_discrete_sampler_draws_fixed_values_for_a_seed():
    # Literal values: a seed that repeats only within one process would pass a comparison of two samplers.
    values = [[0.17], [0.24], [0.5]]
    sampler = _seeded_sampler(DiscreteSamplerCfg(values=values))

    torch.testing.assert_close(
        sampler.sample(8),
        torch.tensor([[0.17], [0.24], [0.24], [0.5], [0.24], [0.17], [0.5], [0.5]]),
    )
    first_draws = _seeded_sampler(DiscreteSamplerCfg(values=values), seed=11).sample(16)
    torch.testing.assert_close(_seeded_sampler(DiscreteSamplerCfg(values=values), seed=11).sample(16), first_draws)


def test_discrete_sampler_draws_from_its_generator_only():
    sampler = _seeded_sampler(DiscreteSamplerCfg(values=MASS_VALUES, probabilities=[0.5, 0.5]))
    global_random_state = torch.get_rng_state()

    sampler.sample(16)

    assert torch.equal(torch.get_rng_state(), global_random_state)


def test_discrete_sampler_follows_probabilities():
    values = [[1.0], [2.0], [3.0]]
    sampler = _seeded_sampler(DiscreteSamplerCfg(values=values, probabilities=[0.2, 0.0, 0.8]), seed=0)
    num_samples = 20000

    samples = sampler.sample(num_samples)[:, 0]

    assert not (samples == 2.0).any(), "An entry with probability 0 must never be drawn."
    assert (samples == 1.0).float().mean().item() == pytest.approx(0.2, abs=0.02)
    assert (samples == 3.0).float().mean().item() == pytest.approx(0.8, abs=0.02)


def test_discrete_sampler_without_probabilities_draws_entries_equally_often():
    sampler = _seeded_sampler(DiscreteSamplerCfg(values=MASS_VALUES), seed=0)

    samples = sampler.sample(20000)[:, 0]

    assert (samples == torch.tensor(0.17)).float().mean().item() == pytest.approx(0.5, abs=0.02)


def test_discrete_sampler_per_env_mode_gives_env_i_entry_i_modulo_the_values():
    values = [[0.17], [0.24], [0.5]]
    sampler = DiscreteSamplerCfg(values=values, random_choice=False).build()
    generator = torch.Generator().manual_seed(7)
    sampler.set_generator(generator)
    generator_state = generator.get_state()

    all_envs = sampler.sample(5, env_ids=torch.arange(5, dtype=torch.int32))
    partial_reset = sampler.sample(2, env_ids=torch.tensor([4, 1]))
    without_env_ids = sampler.sample(4)

    torch.testing.assert_close(all_envs, torch.tensor([[0.17], [0.24], [0.5], [0.17], [0.24]]))
    torch.testing.assert_close(partial_reset, torch.tensor([[0.24], [0.24]]))
    torch.testing.assert_close(without_env_ids, torch.tensor([[0.17], [0.24], [0.5], [0.17]]))
    assert torch.equal(generator.get_state(), generator_state), "The per-env mode must not draw random numbers."


@pytest.mark.parametrize("random_choice", [True, False])
def test_discrete_sampler_replays_recorded_rows_and_notifies_listeners(random_choice: bool):
    sampler = _seeded_sampler(DiscreteSamplerCfg(values=MASS_VALUES, random_choice=random_choice))
    received: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    sampler.add_listener(lambda sample, env_ids: received.append((sample, env_ids)))
    env_ids = torch.tensor([0, 1])

    live = sampler.sample(2, env_ids=env_ids)
    sampler.set_replay_sampler(lambda _count, _env_ids: [[0.24], [0.24]])
    replayed = sampler.sample(2, env_ids=env_ids)

    torch.testing.assert_close(replayed, torch.tensor([[0.24], [0.24]]))
    assert replayed.dtype == torch.float32
    assert len(received) == 2
    assert received[0][0] is live and received[0][1] is env_ids
    assert received[1][0] is replayed and received[1][1] is env_ids


def test_discrete_sampler_falls_back_to_live_draws_when_replay_returns_none():
    sampler = _seeded_sampler(DiscreteSamplerCfg(values=MASS_VALUES, random_choice=False))
    sampler.set_replay_sampler(lambda _count, _env_ids: None)

    torch.testing.assert_close(sampler.sample(2), torch.tensor([[0.17], [0.24]]))


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (DiscreteSamplerCfg(values=[]), "non-empty 'values'"),
        (DiscreteSamplerCfg(values=[0.17, 0.24]), "list of value vectors"),
        (DiscreteSamplerCfg(values=[[0.17], [0.24, 0.3]]), "same non-zero length"),
        (DiscreteSamplerCfg(values=[[], []]), "same non-zero length"),
        (DiscreteSamplerCfg(values=MASS_VALUES, probabilities=[1.0]), "one probability per value"),
        (DiscreteSamplerCfg(values=MASS_VALUES, probabilities=[1.5, -0.5]), "non-negative"),
        (DiscreteSamplerCfg(values=MASS_VALUES, probabilities=[0.5, 0.4]), "sum to 1"),
        (DiscreteSamplerCfg(values=MASS_VALUES, probabilities=[0.5, 0.5], random_choice=False), "random_choice"),
    ],
)
def test_discrete_sampler_rejects_invalid_cfg(cfg: DiscreteSamplerCfg, message: str):
    with pytest.raises(AssertionError, match=message):
        cfg.build()


def test_discrete_sampler_cfg_drives_object_mass_variation():
    cfg = ObjectMassVariationCfg(sampler_cfg=DiscreteSamplerCfg(values=MASS_VALUES, random_choice=False))

    variation = ObjectMassVariation(asset_name="apple", cfg=cfg)
    variation.configure_at_build_time(seed=0, variation_key="apple.mass")
    _, event_cfg = variation.build_event_cfg()

    assert event_cfg.func is ApplyObjectMassFromSampler
    assert isinstance(event_cfg.params["sampler"], DiscreteSampler)
    # ApplyObjectMassFromSampler requires one scalar mass per env.
    assert tuple(event_cfg.params["sampler"].shape_per_sample) == (1,)
    torch.testing.assert_close(
        event_cfg.params["sampler"].sample(3, env_ids=torch.arange(3)), torch.tensor([[0.17], [0.24], [0.17]])
    )
