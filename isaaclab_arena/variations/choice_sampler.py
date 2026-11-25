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


def draw_random_indices(
    num_samples: int,
    num_choices: int,
    generator: torch.Generator | None,
    probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw ``num_samples`` independent indices into a pool of ``num_choices`` items.

    Drawing in several smaller calls from the same generator gives the same indices as one call, so the k-th index does
    not depend on how the draws are split.

    Args:
        num_samples: Number of indices to draw.
        num_choices: Number of items in the pool. Must be positive.
        generator: Generator for the random draws; ``None`` draws from torch's global random state.
        probabilities: Probability of each index, a ``(num_choices,)`` tensor of non-negative values that sum to 1.
            ``None`` draws every index with equal probability.

    Returns:
        An int64 tensor of shape ``(num_samples,)`` with values in ``[0, num_choices)``.
    """
    assert num_choices >= 1, f"num_choices must be positive; got {num_choices}."
    if probabilities is None:
        return torch.randint(low=0, high=num_choices, size=(num_samples,), generator=generator)
    assert probabilities.shape == (
        num_choices,
    ), f"probabilities must have shape ({num_choices},) to match num_choices; got {tuple(probabilities.shape)}."
    # Look up uniform draws in the cumulative distribution. torch.multinomial would draw different indices when the
    # draws are split across calls; torch.rand does not.
    cumulative_probabilities = torch.cumsum(probabilities.to(torch.float64), dim=0)
    uniform_draws = torch.rand(num_samples, generator=generator, dtype=torch.float64) * cumulative_probabilities[-1]
    indices = torch.searchsorted(cumulative_probabilities, uniform_draws, right=True)
    # Rounding can put a draw on the total; map it to the last index that has a non-zero probability.
    last_drawable_index = int(torch.nonzero(probabilities > 0.0).max())
    return indices.clamp(max=last_drawable_index)


@configclass
class ChoiceSamplerCfg(SamplerBaseCfg):
    """Config for ChoiceSampler (empty; ``choices`` is supplied per call)."""

    def build(self) -> ChoiceSampler:
        return ChoiceSampler()


class ChoiceSampler(SamplerBase, Generic[T]):
    """Uniform sampler returning items drawn from a per-call ``choices`` sequence."""

    def sample(self, num_samples: int, choices: Sequence[T], env_ids: torch.Tensor | None = None) -> list[T]:
        """Draw ``num_samples`` items from ``choices``.

        Args:
            num_samples: Number of independent samples to draw, typically the
                number of environments we're drawing a sample for.
            choices: Pool of items to draw from. Must be non-empty.
            env_ids: The env ids the drawn items correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.

        Returns:
            A ``list`` of length ``num_samples`` of items drawn from ``choices``.
        """
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        assert len(choices) >= 1, "ChoiceSampler requires a non-empty 'choices' sequence."
        replay_samples = self._get_replay_samples(num_samples, env_ids)
        if replay_samples is not None:
            assert all(value in choices for value in replay_samples), "Choice replay samples must belong to 'choices'."
            result = replay_samples
        else:
            indices = draw_random_indices(num_samples, len(choices), self._generator)
            result = [choices[int(index)] for index in indices]
        self._notify(result, env_ids)
        return result
