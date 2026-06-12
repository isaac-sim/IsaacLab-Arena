# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from isaaclab.utils import configclass


@configclass
class SamplerBaseCfg:
    """Marker configclass for any sampler cfg.

    Concrete subclasses override :meth:`build` to return their live sampler.
    """

    def build(self) -> SamplerBase:
        """Return the live :class:`SamplerBase` described by this cfg."""
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete SamplerBaseCfg "
            "subclass must provide a build() that returns its live sampler."
        )


class SamplerBase(ABC):
    """Base class shared by every sampler family.

    External observers can subscribe via :meth:`add_listener` to see every value
    drawn; prefer
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`
    so subscriptions survive sampler swaps. Concrete subclasses implement
    :meth:`_sample`; :meth:`sample` wraps it and notifies listeners.
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
        """Draw ``num_samples`` values and notify listeners.

        Delegates to the concrete :meth:`_sample` and then forwards the result to
        every registered listener. Subclasses customize drawing by overriding
        :meth:`_sample`, not this method.

        Args:
            num_samples: Number of independent samples to draw.
            **kwargs: Extra per-call arguments forwarded to :meth:`_sample`
                (e.g. a ``choices`` sequence for a categorical sampler).

        Returns:
            Whatever the concrete sampler produces. Tensor-based samplers return
            a tensor of shape ``(num_samples, *shape_per_sample)``; categorical
            samplers return a ``list`` of length ``num_samples``.
        """
        sample = self._sample(num_samples, **kwargs)
        for listener in self._listeners:
            listener(sample)
        return sample

    @abstractmethod
    def _sample(self, num_samples: int, **kwargs) -> Any:
        """Draw ``num_samples`` values from this distribution."""
        ...
