# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    import torch


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
    """Base class shared by every sampler family."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[Any, torch.Tensor | None], None]] = []

    def add_listener(self, listener: Callable[[Any, torch.Tensor | None], None]) -> None:
        """Register ``listener``, called as ``listener(sample, env_ids)`` for every sample drawn."""
        self._listeners.append(listener)

    def _notify(self, sample: Any, env_ids: torch.Tensor | None = None) -> None:
        """Forward ``sample`` (and the ``env_ids`` it was drawn for) to every registered listener.

        ``env_ids`` is the per-env id tensor/sequence the sample's rows correspond to, or ``None``
        when the single sample applies to all envs (e.g. a build-time draw).
        """
        for listener in self._listeners:
            listener(sample, env_ids)
