# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Samplers for the variation system.

A :class:`Sampler` is a stateless description of how values are drawn. Each
sampler ships a parallel declarative :class:`SamplerCfg` that ``build()`` s
into the live sampler, keeping the Hydra-facing surface as plain data.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import field

from isaaclab.utils import configclass


@configclass
class SamplerCfg:
    """Base configclass for declarative sampler descriptions."""

    def build(self) -> Sampler:
        """Return the live :class:`Sampler` described by this cfg."""
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete SamplerCfg "
            "subclass must provide a build() that returns its live Sampler."
        )


@configclass
class UniformSamplerCfg(SamplerCfg):
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


class Sampler(ABC):
    """Abstract distribution over values.

    Samplers are stateless: they hold distribution parameters but no RNG
    state. External observers can subscribe via :meth:`add_listener` to see
    every value drawn; prefer
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`
    so subscriptions survive sampler swaps.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[torch.Tensor], None]] = []

    def add_listener(self, listener: Callable[[torch.Tensor], None]) -> None:
        """Register ``listener`` to be called with every sample drawn from this sampler.

        Listeners are invoked synchronously inside :meth:`sample`, in registration
        order, with the raw sample tensor (no copy / detach).
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[torch.Tensor], None]) -> None:
        """Remove a previously-registered ``listener``."""
        self._listeners.remove(listener)

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` values from this distribution.

        Args:
            num_samples: Number of independent samples to draw.

        Returns:
            Tensor of shape ``(num_samples, *event_shape)``.
        """
        sample = self._sample(num_samples)
        for listener in self._listeners:
            listener(sample)
        return sample

    @abstractmethod
    def _sample(self, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` values from this distribution."""
        ...


class UniformSampler(Sampler):
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
