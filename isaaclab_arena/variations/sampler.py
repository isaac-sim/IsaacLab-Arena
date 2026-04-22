# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Samplers for the variation system.

A :class:`Sampler` is a stateless description of *how* values are drawn. It
does not own any RNG — instead, callers pass a :class:`torch.Generator` at
sample time so the variation system can control seeding centrally.

Concrete samplers live in this module so they slot in uniformly: a variation
receives a sampler, inspects it if it needs to translate it for a foreign
API (see e.g. :class:`~isaaclab_arena.variations.object_color.ObjectColorVariation`),
or just calls :meth:`Sampler.sample` on it at event time.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence


class Sampler(ABC):
    """Abstract distribution over values.

    Implementations are expected to be stateless: all randomness flows
    through the ``generator`` argument, so repeated sampling with the same
    generator state is reproducible.
    """

    @abstractmethod
    def sample(self, num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``num_samples`` values from this distribution.

        Args:
            num_samples: Number of independent samples to draw. Must be ``>= 0``.
            generator: Optional generator to pull randomness from. If ``None``,
                the default torch RNG is used.

        Returns:
            Tensor of shape ``(num_samples, *event_shape)`` where
            ``event_shape`` is empty for scalar samplers and ``(d,)`` for
            vector-valued samplers (e.g. an RGB uniform).
        """
        ...


class UniformSampler(Sampler):
    """Uniform sampler over ``[low, high]``.

    ``low`` and ``high`` may be scalars (producing scalar samples) or
    broadcast-compatible sequences (producing vector samples of the same
    shape). The bounds are inclusive; samples are drawn with
    ``low + (high - low) * U(0, 1)``.

    Examples:
        Scalar mass range::

            UniformSampler(0.1, 1.0)

        Per-channel RGB range::

            UniformSampler(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0))
    """

    def __init__(self, low: float | Sequence[float], high: float | Sequence[float]):
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
        """Shape of a single sample (empty for scalar samplers)."""
        return self.low.shape

    def sample(self, num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        shape = (num_samples, *self.event_shape)
        u = torch.rand(shape, generator=generator)
        return self.low + (self.high - self.low) * u
