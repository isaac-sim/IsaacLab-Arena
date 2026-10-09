# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check deterministic choice sampling and replay behavior."""

import torch

import pytest

from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.sequential_choice_sampler import SequentialChoiceSampler, SequentialChoiceSamplerCfg


def test_sequential_choices_restart_each_call_and_notify_listeners():
    sampler = SequentialChoiceSamplerCfg().build()
    received_samples = []
    sampler.add_listener(lambda sample, env_ids: received_samples.append((sample, env_ids)))
    env_ids = torch.tensor([8, 2, 5, 3, 6])

    first_draw = sampler.sample(5, choices=["banana", "orange"], env_ids=env_ids)
    assert first_draw == ["banana", "orange", "banana", "orange", "banana"]
    assert sampler.sample(3, choices=["banana", "orange"]) == ["banana", "orange", "banana"]
    assert received_samples[0][0] is first_draw
    assert received_samples[0][1] is env_ids
    assert len(received_samples) == 2
    assert sampler.sample(0, choices=["banana"]) == []


def test_choice_generator_is_reproducible_and_independent_of_global_random_state():
    sampler = ChoiceSampler()
    generator = torch.Generator().manual_seed(42)
    original_global_state = torch.random.get_rng_state()
    first_draw = sampler.sample(20, choices=range(100), generator=generator)
    assert torch.equal(torch.random.get_rng_state(), original_global_state)

    with torch.random.fork_rng(devices=[]):
        torch.rand(15)
        repeated_draw = sampler.sample(20, choices=range(100), generator=torch.Generator().manual_seed(42))
    assert repeated_draw == first_draw


@pytest.mark.parametrize("sampler_type", [ChoiceSampler, SequentialChoiceSampler])
def test_choice_replay_notifies_without_advancing_generator(sampler_type):
    sampler = sampler_type()
    env_ids = torch.tensor([4, 1])
    replay_requests = []
    received_samples = []

    def replay_samples(num_samples, requested_env_ids):
        replay_requests.append((num_samples, requested_env_ids))
        return ["orange", "orange"]

    sampler.set_replay_sampler(replay_samples)
    sampler.add_listener(lambda sample, ids: received_samples.append((sample, ids)))
    generator = torch.Generator().manual_seed(11)
    original_generator_state = generator.get_state()
    samples = sampler.sample(2, choices=["banana", "orange"], env_ids=env_ids, generator=generator)

    assert samples == ["orange", "orange"]
    assert replay_requests[0][0] == 2
    assert replay_requests[0][1] is env_ids
    assert received_samples[0][0] is samples
    assert received_samples[0][1] is env_ids
    assert torch.equal(generator.get_state(), original_generator_state)


def test_sequential_choice_replay_preserves_validation():
    sampler = SequentialChoiceSampler()
    sampler.set_replay_sampler(lambda _count, _env_ids: ["orange"])
    with pytest.raises(AssertionError, match="non-negative"):
        sampler.sample(-1, choices=["orange"])
    with pytest.raises(AssertionError, match="non-empty"):
        sampler.sample(1, choices=[])
    with pytest.raises(AssertionError, match="must belong"):
        sampler.sample(1, choices=["banana"])
    with pytest.raises(AssertionError, match="1 rows for a 2-sample draw"):
        sampler.sample(2, choices=["orange"])
