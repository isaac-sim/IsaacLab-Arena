# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Generic, TypeVar

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

T = TypeVar("T")


@configclass
class ChoiceSamplerCfg(SamplerBaseCfg):
    """Config for ChoiceSampler."""

    distribution: str = "choice"
    """Selects this sampler in a variation override: ``<variation>.sampler_cfg.distribution=choice``."""

    values: list[Any] = field(default_factory=list)
    """Entries to draw from, for example ``[[0.17], [0.24]]`` for two object masses. Leave empty for a variation that
    passes its own entries to each draw, such as the HDR names of ``HDRImageVariation``."""

    probabilities: list[float] | None = None
    """Probability of drawing each entry: non-negative, summing to 1. ``None`` draws every entry with equal
    probability. Only valid when ``random_choice`` is true."""

    random_choice: bool = True
    """True draws an entry at random on every draw, seeded by the variation's generator. False draws no random numbers:
    env ``i`` gets entry ``i % len(entries)`` on every draw. A draw without env ids counts them from 0, so a build-time
    variation always gets the first entry."""

    def build(self) -> ChoiceSampler:
        return ChoiceSampler(values=self.values, probabilities=self.probabilities, random_choice=self.random_choice)


class ChoiceSampler(SamplerBase, Generic[T]):
    """Sampler returning entries of a fixed set, drawn at random or as one fixed entry per env.

    The entries are the sampler's ``values``, or the ``choices`` passed to each draw when ``values`` is empty.

    Args:
        values: Entries to draw from. Empty when each draw passes its own ``choices``.
        probabilities: Probability of each entry, or ``None`` for equal probabilities. Only valid with
            ``random_choice``.
        random_choice: Whether to draw entries at random, or give env ``i`` the entry ``i % len(entries)``.
    """

    def __init__(
        self,
        values: Sequence[T] = (),
        probabilities: Sequence[float] | None = None,
        random_choice: bool = True,
    ):
        super().__init__()
        self.values: list[T] = list(values)
        self.random_choice = random_choice
        self.probabilities: torch.Tensor | None = None
        if probabilities is not None:
            assert random_choice, "ChoiceSampler 'probabilities' apply only when random_choice is true."
            assert not self.values or len(probabilities) == len(self.values), (
                f"ChoiceSampler needs one probability per value; got {len(probabilities)} probabilities for"
                f" {len(self.values)} values."
            )
            self.probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
            assert torch.all(
                self.probabilities >= 0.0
            ), f"ChoiceSampler probabilities must be non-negative; got {list(probabilities)}."
            probability_sum = float(self.probabilities.sum())
            assert math.isclose(
                probability_sum, 1.0, abs_tol=1e-6
            ), f"ChoiceSampler probabilities must sum to 1; got {list(probabilities)} with sum {probability_sum}."

    def sample(
        self,
        num_samples: int,
        choices: Sequence[T] | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> list[T]:
        """Draw ``num_samples`` entries.

        Args:
            num_samples: Number of samples to draw, typically the number of environments we're drawing a sample for.
            choices: Entries to draw from in this call. Required when the sampler has no ``values``, and not allowed
                when it has.
            env_ids: The env ids the drawn entries correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.

        Returns:
            A ``list`` of length ``num_samples`` of entries.
        """
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        if choices is None:
            entries = self.values
        else:
            assert not self.values, (
                f"ChoiceSampler got 'choices' for a draw, but it has its own values {self.values}. Leave the sampler's"
                " values empty for a variation that supplies the choices."
            )
            entries = list(choices)
        assert len(entries) >= 1, "ChoiceSampler requires a non-empty 'values' list or 'choices' sequence."
        if self.probabilities is not None:
            assert len(self.probabilities) == len(entries), (
                f"ChoiceSampler needs one probability per entry; got {len(self.probabilities)} probabilities for"
                f" {len(entries)} entries."
            )

        replay_samples = self._get_replay_samples(num_samples, env_ids)
        if replay_samples is not None:
            assert all(value in entries for value in replay_samples), "Choice replay samples must belong to 'choices'."
            result = replay_samples
        else:
            if self.random_choice:
                indices = self._draw_random_indices(num_samples, len(entries))
            else:
                indices = self._env_ids_to_entry_indices(num_samples, len(entries), env_ids)
            result = [entries[int(index)] for index in indices]
        self._notify(result, env_ids)
        return result

    def _draw_random_indices(self, num_samples: int, num_entries: int) -> torch.Tensor:
        """Draw ``num_samples`` independent indices into ``num_entries`` entries, weighted by ``self.probabilities``.

        Drawing in several smaller calls from the same generator gives the same indices as one call.
        """
        if self.probabilities is None:
            return torch.randint(low=0, high=num_entries, size=(num_samples,), generator=self._generator)
        # Look up uniform draws in the cumulative distribution. torch.multinomial would draw different indices when the
        # draws are split across calls; torch.rand does not.
        cumulative_probabilities = torch.cumsum(self.probabilities, dim=0)
        uniform_draws = torch.rand(num_samples, generator=self._generator, dtype=torch.float64)
        indices = torch.searchsorted(cumulative_probabilities, uniform_draws * cumulative_probabilities[-1], right=True)
        # Rounding can put a draw on the total; map it to the last index that has a non-zero probability.
        last_drawable_index = int(torch.nonzero(self.probabilities > 0.0).max())
        return indices.clamp(max=last_drawable_index)

    def _env_ids_to_entry_indices(
        self, num_samples: int, num_entries: int, env_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the entry index for each env, ``env_id % num_entries``; ``None`` env ids count from 0."""
        if env_ids is None:
            env_ids = torch.arange(num_samples)
        env_ids_cpu = torch.as_tensor(env_ids, dtype=torch.int64, device="cpu").reshape(-1)
        assert (
            len(env_ids_cpu) == num_samples
        ), f"ChoiceSampler got {len(env_ids_cpu)} env ids for a {num_samples}-sample draw."
        return env_ids_cpu % num_entries
