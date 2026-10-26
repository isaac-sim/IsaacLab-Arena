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
    """Sampler returning consecutive items of a per-call ``choices`` sequence, starting over after the last.

    It draws no random numbers, so the generator does not change its items.
    """

    def sample(
        self,
        num_samples: int,
        choices: Sequence[T],
        *,
        start_index: int,
        env_ids: torch.Tensor | None = None,
    ) -> list[T]:
        """Return ``num_samples`` consecutive items of ``choices``, starting at ``start_index``.

        When a replay sampler supplies rows, returns those rows instead.

        Args:
            num_samples: Number of items to return, typically the number of environments they are for.
            choices: Items to take in list order. Must be non-empty.
            start_index: Non-negative index of the first item. An index past the last item wraps around, so the
                first item is ``choices[start_index % len(choices)]``.
            env_ids: The env ids the items correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.

        Returns:
            A ``list`` of length ``num_samples`` of items from ``choices``.
        """
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        assert len(choices) >= 1, "SequentialSampler requires a non-empty 'choices' sequence."
        assert start_index >= 0, f"start_index must be non-negative; got {start_index}."
        replay_samples = self._get_replay_samples(num_samples, env_ids)
        if replay_samples is not None:
            values_outside_choices = [value for value in replay_samples if value not in choices]
            assert (
                not values_outside_choices
            ), f"Sequential replay samples {values_outside_choices} must belong to 'choices': {list(choices)}."
            result = replay_samples
        else:
            result = [choices[(start_index + offset) % len(choices)] for offset in range(num_samples)]
        self._notify(result, env_ids)
        return result
