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

Each sampler ships a parallel declarative cfg (:class:`SamplerCfg` +
subclasses) with a :meth:`~SamplerCfg.build` factory. The cfgs are what the
variation cfg (and therefore Hydra overrides) sees — the live :class:`Sampler`
instance is built from the cfg at variation-construction time. This keeps the
Hydra surface "plain data" (lists of floats) while the runtime side keeps
torch tensors and other non-serialisable bits.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import field

from isaaclab.utils import configclass


@configclass
class SamplerCfg:
    """Base configclass for declarative sampler descriptions.

    Concrete sampler cfgs subclass this and add their parameters plus a
    :meth:`build` factory that returns the live :class:`Sampler`. The base
    declares :meth:`build` as an abstract (unimplemented) method so that
    downstream code — e.g.
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.set_sampler`
    — can rely on every sampler cfg being build-able without type-narrowing
    to a specific subclass.
    """

    def build(self) -> Sampler:
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete SamplerCfg "
            "subclass must provide a build() that returns its live Sampler."
        )


@configclass
class UniformSamplerCfg(SamplerCfg):
    """Config for :class:`UniformSampler`.

    ``low`` and ``high`` are kept as ``list[float]`` (not tuple) so that
    OmegaConf / Hydra round-trips cleanly — OmegaConf canonicalises any
    sequence override to a :class:`omegaconf.ListConfig`. The live
    :class:`UniformSampler` stores them as torch tensors and performs the
    shape / ordering validation on construction.

    Attributes:
        low: Lower bound per dimension. Length determines the sampler's
            ``event_shape``; ``[0.0]`` is a scalar uniform, ``[0.0, 0.0, 0.0]``
            a 3D uniform (e.g. RGB).
        high: Upper bound per dimension. Must have the same length as ``low``
            and be element-wise ``>= low`` — both checks are enforced by
            :class:`UniformSampler`'s ctor at build time.
    """

    low: list[float] = field(default_factory=lambda: [0.0])
    high: list[float] = field(default_factory=lambda: [1.0])

    def build(self) -> UniformSampler:
        return UniformSampler(low=self.low, high=self.high)


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
