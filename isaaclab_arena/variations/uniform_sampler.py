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
