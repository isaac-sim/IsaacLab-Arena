# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChoiceSampler draws at a given index."""

import torch

import pytest

from isaaclab_arena.variations.choice_sampler import ChoiceSampler


def test_choice_sampler_returns_the_choice_at_index():
    indexed_sampler = ChoiceSampler()
    received_samples = []
    indexed_sampler.add_listener(lambda sample, env_ids: received_samples.append((sample, env_ids)))

    samples = indexed_sampler.sample(3, choices=["a", "b", "c"], index=2)

    assert samples == ["c", "c", "c"]
    assert received_samples == [(["c", "c", "c"], None)]

    replaying_sampler = ChoiceSampler()
    replaying_sampler.set_replay_sampler(lambda num_samples, env_ids: ["b"])
    assert replaying_sampler.sample(1, choices=["a", "b"], index=0) == ["b"]

    with pytest.raises(AssertionError, match="not both"):
        indexed_sampler.sample(1, choices=["a", "b"], generator=torch.Generator(), index=0)
