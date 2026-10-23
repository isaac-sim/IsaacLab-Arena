# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Generic, TypeVar

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

T = TypeVar("T")


@configclass
class SequentialSamplerCfg(SamplerBaseCfg):
    """Config for SequentialSampler (empty; ``choices`` is supplied per call)."""

    def build(self) -> SequentialSampler:
        return SequentialSampler()


class SequentialSampler(SamplerBase, Generic[T]):
    """Sampler returning the items of a per-call ``choices`` sequence in list order, starting over after the last.

    It draws no random numbers, so the generator does not change its items.
    """

    def __init__(self) -> None:
        super().__init__()
        self._position = 0

    def set_position(self, position: int) -> None:
        """Set the index in ``choices`` of the item the next draw returns first.

        Args:
            position: Non-negative index. A position past the last item wraps around, so the next draw starts at
                ``choices[position % len(choices)]``.
        """
        assert position >= 0, f"position must be non-negative; got {position}."
        self._position = position

    def sample(self, num_samples: int, choices: Sequence[T], env_ids: torch.Tensor | None = None) -> list[T]:
        """Return the next ``num_samples`` items of ``choices`` and advance the position past them.

        When a replay sampler supplies rows, returns those rows and leaves the position unchanged.

        Args:
            num_samples: Number of items to return, typically the number of environments they are for.
            choices: Items to take in list order. Must be non-empty.
            env_ids: The env ids the items correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.

        Returns:
            A ``list`` of length ``num_samples`` of items from ``choices``.
        """
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        assert len(choices) >= 1, "SequentialSampler requires a non-empty 'choices' sequence."
        replay_samples = self._get_replay_samples(num_samples, env_ids)
        if replay_samples is not None:
            values_outside_choices = [value for value in replay_samples if value not in choices]
            assert (
                not values_outside_choices
            ), f"Sequential replay samples {values_outside_choices} must belong to 'choices': {list(choices)}."
            result = replay_samples
        else:
            result = [choices[(self._position + offset) % len(choices)] for offset in range(num_samples)]
            self._position = (self._position + num_samples) % len(choices)
        self._notify(result, env_ids)
        return result
