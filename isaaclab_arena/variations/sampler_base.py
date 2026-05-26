# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sampler abstract base for the variation system.

A :class:`SamplerBase` is a stateless description of how values are drawn.
Each sampler ships a parallel declarative :class:`SamplerBaseCfg` that
``build()`` s into the live sampler, keeping the Hydra-facing surface as
plain data.

The return type of :meth:`SamplerBase.sample` is intentionally loose
(``Any``): continuous samplers (e.g. :class:`UniformSampler`) return a
``torch.Tensor`` of shape ``(num_samples, *event_shape)``, while categorical
samplers return a ``list`` of items drawn from a per-call ``choices``
sequence. Listeners (e.g. :class:`~isaaclab_arena.variations.variations_recorder.VariationRecorder`)
should handle either shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from isaaclab.utils import configclass


@configclass
class SamplerBaseCfg:
    """Base configclass for declarative sampler descriptions."""

    def build(self) -> SamplerBase:
        """Return the live :class:`SamplerBase` described by this cfg."""
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete SamplerBaseCfg "
            "subclass must provide a build() that returns its live SamplerBase."
        )


class SamplerBase(ABC):
    """Abstract distribution over values.

    Samplers are stateless: they hold distribution parameters but no RNG
    state. External observers can subscribe via :meth:`add_listener` to see
    every value drawn; prefer
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`
    so subscriptions survive sampler swaps.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[Any], None]] = []

    def add_listener(self, listener: Callable[[Any], None]) -> None:
        """Register ``listener`` to be called with every sample drawn from this sampler.

        Listeners are invoked synchronously inside :meth:`sample`, in registration
        order, with the raw sample value (no copy / detach).
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Any], None]) -> None:
        """Remove a previously-registered ``listener``."""
        self._listeners.remove(listener)

    def sample(self, num_samples: int, **kwargs) -> Any:
        """Draw ``num_samples`` values from this distribution.

        Args:
            num_samples: Number of independent samples to draw.
            **kwargs: Extra per-call arguments passed through to
                :meth:`_sample`. Concrete samplers use this for parameters
                that aren't part of the distribution's declarative cfg
                (e.g. a list of choices for a categorical sampler).

        Returns:
            Whatever the concrete sampler produces. Tensor-based samplers
            return a tensor of shape ``(num_samples, *event_shape)``;
            categorical samplers return a ``list`` of length ``num_samples``.
        """
        sample = self._sample(num_samples, **kwargs)
        for listener in self._listeners:
            listener(sample)
        return sample

    @abstractmethod
    def _sample(self, num_samples: int, **kwargs) -> Any:
        """Draw ``num_samples`` values from this distribution."""
        ...
