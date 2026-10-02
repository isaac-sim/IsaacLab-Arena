# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import field

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.continuous_sampler import ContinuousSampler, ContinuousSamplerCfg


@configclass
class UniformSamplerCfg(ContinuousSamplerCfg):
    """Config for :class:`UniformSampler`."""

    low: list[float] = field(default_factory=lambda: [0.0])
    """Lower bound per dimension. Length determines the sampler's shape_per_sample."""

    high: list[float] = field(default_factory=lambda: [1.0])
    """Upper bound per dimension. Same length as ``low``, element-wise ``>= low``."""

    def build(self) -> UniformSampler:
        return UniformSampler(low=self.low, high=self.high)


class UniformSampler(ContinuousSampler):
    """Uniform sampler over ``[low, high]``.

    ``low`` and ``high`` may be scalars or broadcast-compatible sequences;
    samples are drawn with ``low + (high - low) * U(0, 1)``.
    """

    def __init__(self, low: float | Sequence[float], high: float | Sequence[float]):
        super().__init__()
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        assert (
            self.low.shape == self.high.shape
        ), f"UniformSampler low/high must have matching shape; got {tuple(self.low.shape)} vs {tuple(self.high.shape)}."
        assert torch.isfinite(self.low).all() and torch.isfinite(self.high).all(), "Uniform bounds must be finite."
        assert torch.all(
            self.low <= self.high
        ), f"UniformSampler requires low <= high elementwise; got low={self.low}, high={self.high}."

    @property
    def shape_per_sample(self) -> torch.Size:
        return self.low.shape

    def _sample(self, num_samples: int) -> torch.Tensor:
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        shape = (num_samples, *self.shape_per_sample)
        u = torch.rand(shape)
        return self.low + (self.high - self.low) * u

    def _sample_with_generator(self, generator: torch.Generator) -> torch.Tensor:
        u = torch.rand(self.shape_per_sample, generator=generator)
        return self.low + (self.high - self.low) * u

    def _replay_value(self, value) -> torch.Tensor:
        result = super()._replay_value(value)
        assert (
            (result >= self.low) & (result <= self.high)
        ).all(), f"Replay value for '{self._variation_path}' is outside the configured uniform bounds."
        return result

    def validate_range(
        self,
        shape: tuple[int, ...],
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> None:
        """Check that every possible draw satisfies a variation's physical domain."""
        super().validate_range(shape, minimum=minimum, maximum=maximum, minimum_inclusive=minimum_inclusive)
        if minimum is not None:
            valid = self.low >= minimum if minimum_inclusive else self.low > minimum
            assert valid.all(), f"Sampler lower bounds violate the physical minimum {minimum}."
        if maximum is not None:
            assert (self.high <= maximum).all(), f"Sampler upper bounds violate the physical maximum {maximum}."
