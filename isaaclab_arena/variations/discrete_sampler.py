# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import field

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import draw_random_indices
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler, ContinuousSamplerCfg


@configclass
class DiscreteSamplerCfg(ContinuousSamplerCfg):
    """Config for :class:`DiscreteSampler`."""

    distribution: str = "discrete"
    """Selects this sampler in a variation override: ``<variation>.sampler_cfg.distribution=discrete``."""

    values: list[list[float]] = field(default_factory=list)
    """Value vectors to take samples from, for example ``[[0.17], [0.24]]`` for two scalar values. Required. Every
    vector has the same length, which sets the sampler's shape_per_sample."""

    probabilities: list[float] | None = None
    """Probability of drawing each entry of ``values``: non-negative, summing to 1. ``None`` draws every entry with
    equal probability. Only valid when ``random_choice`` is true."""

    random_choice: bool = True
    """True draws an entry at random on every draw, seeded by the variation's generator. False draws no random numbers:
    env ``i`` gets ``values[i % len(values)]`` on every reset. A draw without env ids counts them from 0, so a
    build-time variation always gets ``values[0]``."""

    def build(self) -> DiscreteSampler:
        return DiscreteSampler(values=self.values, probabilities=self.probabilities, random_choice=self.random_choice)


class DiscreteSampler(ContinuousSampler):
    """Sampler returning value vectors taken from a fixed list, as float32 tensors like :class:`UniformSampler`'s.

    Args:
        values: Value vectors to take samples from. Must be non-empty, with every vector of the same non-zero length.
        probabilities: Probability of each entry of ``values``, or ``None`` for equal probabilities. Only valid
            with ``random_choice``.
        random_choice: Whether to draw entries at random, or give env ``i`` the entry ``values[i % len(values)]``.
    """

    def __init__(
        self,
        values: Sequence[Sequence[float]],
        probabilities: Sequence[float] | None = None,
        random_choice: bool = True,
    ):
        super().__init__()
        assert len(values) >= 1, "DiscreteSampler requires a non-empty 'values' list, for example [[0.17], [0.24]]."
        value_lengths = [len(value) if isinstance(value, Sequence) else None for value in values]
        assert (
            None not in value_lengths
        ), f"DiscreteSampler 'values' must be a list of value vectors, for example [[0.17], [0.24]]; got {values}."
        assert (
            len(set(value_lengths)) == 1 and value_lengths[0] >= 1
        ), f"DiscreteSampler value vectors must all have the same non-zero length; got lengths {value_lengths}."
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.random_choice = random_choice
        self.probabilities: torch.Tensor | None = None
        if probabilities is not None:
            assert random_choice, "DiscreteSampler 'probabilities' apply only when random_choice is true."
            assert len(probabilities) == len(values), (
                f"DiscreteSampler needs one probability per value; got {len(probabilities)} probabilities for"
                f" {len(values)} values."
            )
            self.probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
            assert torch.all(
                self.probabilities >= 0.0
            ), f"DiscreteSampler probabilities must be non-negative; got {list(probabilities)}."
            probability_sum = float(self.probabilities.sum())
            assert math.isclose(
                probability_sum, 1.0, abs_tol=1e-6
            ), f"DiscreteSampler probabilities must sum to 1; got {list(probabilities)} with sum {probability_sum}."

    @property
    def shape_per_sample(self) -> torch.Size:
        return self.values.shape[1:]

    def _sample(self, num_samples: int, env_ids: torch.Tensor | None) -> torch.Tensor:
        if self.random_choice:
            indices = draw_random_indices(num_samples, len(self.values), self._generator, self.probabilities)
        else:
            indices = self._env_ids_to_value_indices(num_samples, env_ids)
        return self.values[indices]

    def _env_ids_to_value_indices(self, num_samples: int, env_ids: torch.Tensor | None) -> torch.Tensor:
        """Return the index into ``values`` for each env, ``env_id % len(values)``; ``None`` env ids count from 0."""
        if env_ids is None:
            env_ids = torch.arange(num_samples)
        env_ids_cpu = torch.as_tensor(env_ids, dtype=torch.int64, device="cpu").reshape(-1)
        assert (
            len(env_ids_cpu) == num_samples
        ), f"DiscreteSampler got {len(env_ids_cpu)} env ids for a {num_samples}-sample draw."
        return env_ids_cpu % len(self.values)
