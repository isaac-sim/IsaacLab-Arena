# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChoiceSampler draws in order from a start index."""

import torch

import pytest

from isaaclab_arena.variations.choice_sampler import ChoiceSampler


def test_choice_sampler_takes_choices_in_order_from_start_index():
    in_order_sampler = ChoiceSampler()
    received_samples = []
    in_order_sampler.add_listener(lambda sample, env_ids: received_samples.append((sample, env_ids)))

    samples = in_order_sampler.sample(5, choices=["a", "b", "c"], start_index=2)

    assert samples == ["c", "a", "b", "c", "a"]
    assert received_samples == [(["c", "a", "b", "c", "a"], None)]

    replaying_sampler = ChoiceSampler()
    replaying_sampler.set_replay_sampler(lambda num_samples, env_ids: ["b"])
    assert replaying_sampler.sample(1, choices=["a", "b"], start_index=0) == ["b"]

    with pytest.raises(AssertionError, match="not both"):
        in_order_sampler.sample(1, choices=["a", "b"], generator=torch.Generator(), start_index=0)
