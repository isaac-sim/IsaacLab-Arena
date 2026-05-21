# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Uniform distribution sampler over ``[low, high]``."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import field

from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


@configclass
class UniformSamplerCfg(SamplerBaseCfg):
    """Config for :class:`UniformSampler`.

    Attributes:
        low: Lower bound per dimension. Length determines the sampler's
            ``event_shape``.
        high: Upper bound per dimension. Must have the same length as ``low``
            and be element-wise ``>= low``.
    """

    low: list[float] = field(default_factory=lambda: [0.0])
    high: list[float] = field(default_factory=lambda: [1.0])

    def build(self) -> UniformSampler:
        return UniformSampler(low=self.low, high=self.high)


class UniformSampler(SamplerBase):
    """Uniform sampler over ``[low, high]``.

    ``low`` and ``high`` may be scalars or broadcast-compatible sequences;
    samples are drawn with ``low + (high - low) * U(0, 1)``.
    """

    def __init__(self, low: float | Sequence[float], high: float | Sequence[float]):
        super().__init__()
        low_t = torch.as_tensor(low, dtype=torch.float32)
        high_t = torch.as_tensor(high, dtype=torch.float32)
        assert (
            low_t.shape == high_t.shape
        ), f"UniformSampler low/high must have matching shape; got {tuple(low_t.shape)} vs {tuple(high_t.shape)}."
        assert torch.all(
            low_t <= high_t
        ), f"UniformSampler requires low <= high elementwise; got low={low_t}, high={high_t}."
        self.low = low_t
        self.high = high_t

    @property
    def event_shape(self) -> torch.Size:
        """Shape of a single sample."""
        return self.low.shape

    def _sample(self, num_samples: int) -> torch.Tensor:
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        shape = (num_samples, *self.event_shape)
        u = torch.rand(shape)
        return self.low + (self.high - self.low) * u
