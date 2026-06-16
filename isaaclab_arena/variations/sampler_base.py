# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
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

    Observers subscribe via ``add_listener`` to see every value drawn; prefer
    ``VariationBase.add_sample_listener`` so subscriptions survive sampler swaps. Each sampler
    family declares its own typed ``sample``; that method draws the value and forwards it to
    listeners via ``_notify``.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[Any], None]] = []

    def add_listener(self, listener: Callable[[Any], None]) -> None:
        """Register ``listener`` to be called with every sample drawn from this sampler.

        Listeners are invoked synchronously inside ``sample``, in registration order, with
        the raw sample value (no copy / detach).
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Any], None]) -> None:
        """Remove a previously-registered ``listener``."""
        self._listeners.remove(listener)

    def _notify(self, sample: Any) -> None:
        """Forward ``sample`` to every registered listener, in registration order."""
        for listener in self._listeners:
            listener(sample)
