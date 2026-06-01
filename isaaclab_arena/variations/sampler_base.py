# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sampler abstract base for the variation system.

A :class:`SamplerBase` is a stateless description of how values are drawn,
paired with a declarative :class:`SamplerBaseCfg` that ``build()`` s into it.

:meth:`SamplerBase.sample` returns ``Any``: continuous samplers return a
``torch.Tensor`` of shape ``(num_samples, *event_shape)``; categorical samplers
return a ``list``. Listeners should handle either shape.
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

    Stateless: holds distribution parameters but no RNG state. Subscribe via
    :meth:`add_listener` to observe every value drawn, or
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`
    to survive sampler swaps.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[Any], None]] = []

    def add_listener(self, listener: Callable[[Any], None]) -> None:
        """Register ``listener`` to receive every sample drawn from this sampler.

        Called synchronously inside :meth:`sample` with the raw value (no copy).
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Any], None]) -> None:
        """Remove a previously-registered ``listener``."""
        self._listeners.remove(listener)

    def sample(self, num_samples: int, **kwargs) -> Any:
        """Draw ``num_samples`` values, notify listeners, and return them.

        Args:
            num_samples: Number of independent samples to draw.
            **kwargs: Extra per-call arguments forwarded to the concrete sampler
                (e.g. ``choices`` for a categorical sampler).

        Returns:
            A tensor of shape ``(num_samples, *event_shape)`` for tensor
            samplers, or a ``list`` of length ``num_samples`` for categorical
            samplers.
        """
        sample = self._sample(num_samples, **kwargs)
        for listener in self._listeners:
            listener(sample)
        return sample

    @abstractmethod
    def _sample(self, num_samples: int, **kwargs) -> Any:
        """Draw ``num_samples`` values from this distribution."""
        ...
