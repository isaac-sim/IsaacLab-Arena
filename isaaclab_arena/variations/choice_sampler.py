# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
class ChoiceSamplerCfg(SamplerBaseCfg):
    """Config for ChoiceSampler (empty; ``choices`` is supplied per call)."""

    def build(self) -> ChoiceSampler:
        return ChoiceSampler()


class ChoiceSampler(SamplerBase, Generic[T]):
    """Sampler returning items from a per-call ``choices`` sequence, drawn at random or in order from a start index."""

    def sample(
        self,
        num_samples: int,
        choices: Sequence[T],
        env_ids: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        start_index: int | None = None,
    ) -> list[T]:
        """Draw ``num_samples`` items from ``choices``.

        Args:
            num_samples: Number of items to return, typically the
                number of environments we're drawing a sample for.
            choices: Pool of items to draw from. Must be non-empty.
            env_ids: The env ids the drawn items correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.
            generator: Generator to draw from; ``None`` draws from torch's global random state.
                Unused when a replay sampler supplies the rows.
            start_index: Index of the first item to take; the following items follow ``choices`` in order,
                wrapping around. ``None`` draws at random. Unused when a replay sampler supplies the rows.

        Returns:
            A ``list`` of length ``num_samples`` of items drawn from ``choices``.
        """
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        assert len(choices) >= 1, "ChoiceSampler requires a non-empty 'choices' sequence."
        assert generator is None or start_index is None, "Pass either generator or start_index, not both."
        replay_samples = self._get_replay_samples(num_samples, env_ids)
        if replay_samples is not None:
            assert all(value in choices for value in replay_samples), "Choice replay samples must belong to 'choices'."
            result = replay_samples
        elif start_index is not None:
            result = [choices[(start_index + offset) % len(choices)] for offset in range(num_samples)]
        else:
            indices = torch.randint(low=0, high=len(choices), size=(num_samples,), generator=generator)
            result = [choices[int(index)] for index in indices]
        self._notify(result, env_ids)
        return result
