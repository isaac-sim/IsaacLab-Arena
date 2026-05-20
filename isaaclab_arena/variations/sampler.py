# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Samplers for the variation system.

A :class:`Sampler` is a stateless description of *how* values are drawn.
Randomness currently comes from torch's default global RNG; per-sampler
generator threading was removed for now because no caller was passing one
in. It can be re-added once the variation system grows a real seeding
story (per-env / per-episode generators) and there is somewhere to plumb
one through.

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
from collections.abc import Callable, Sequence
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

    Implementations are expected to be stateless — they hold the
    parameters of the distribution (e.g. ``low`` / ``high``) but no RNG
    state. Randomness is currently pulled from torch's default global
    generator; see the module docstring for why explicit generator
    threading was removed.

    Sample observation:
        :class:`Sampler` exposes a small listener API
        (:meth:`add_listener`, :meth:`remove_listener`) so external code
        — typically the variation system's recording layer
        (:class:`~isaaclab_arena.variations.ledger.VariationLedger`) —
        can observe every value drawn by :meth:`sample`. The public way
        to subscribe is through
        :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`,
        which keeps the listener bookkeeping correctly bound to the
        variation across :meth:`~isaaclab_arena.variations.variation_base.VariationBase.set_sampler`
        swaps; reaching past the variation to call
        :meth:`Sampler.add_listener` directly is supported but only
        binds to *this* sampler instance — a subsequent
        ``set_sampler(...)`` on the owning variation will drop the
        listener.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[torch.Tensor], None]] = []

    def add_listener(self, listener: Callable[[torch.Tensor], None]) -> None:
        """Register ``listener`` to be called with every sample drawn from this sampler.

        Listeners are invoked synchronously inside :meth:`sample`, in
        registration order, after the underlying :meth:`_sample` call
        returns. They receive the raw sample tensor as-is (no copy /
        detach) — observers that want to retain it across timesteps
        should detach / clone themselves.

        Prefer subscribing via
        :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`
        unless you have a specific reason to bind to a single sampler
        instance — see this class's docstring for the trade-off.
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[torch.Tensor], None]) -> None:
        """Remove ``listener`` from this sampler.

        Raises:
            ValueError: If ``listener`` is not currently registered. The
                caller is expected to track its own registrations; we
                fail loudly rather than silently ignoring unknown
                listeners so bookkeeping bugs surface early.
        """
        self._listeners.remove(listener)

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` values from this distribution.

        Args:
            num_samples: Number of independent samples to draw. Must be ``>= 0``.

        Returns:
            Tensor of shape ``(num_samples, *event_shape)`` where
            ``event_shape`` is empty for scalar samplers and ``(d,)`` for
            vector-valued samplers (e.g. an RGB uniform).
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
        """Shape of a single sample (empty for scalar samplers)."""
        return self.low.shape

    def _sample(self, num_samples: int) -> torch.Tensor:
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        shape = (num_samples, *self.event_shape)
        u = torch.rand(shape)
        return self.low + (self.high - self.low) * u
