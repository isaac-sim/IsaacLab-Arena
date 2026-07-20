# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base classes.

A :class:`VariationBase` pairs a target asset with a sampler and a hook that
realises one tweak to the scene. Variations attach to any
:class:`~isaaclab_arena.assets.asset.Asset` and start disabled. Concrete
variations subclass one of two flavors:

* :class:`RunTimeVariationBase` — realised via an event term during simulation
  (e.g. per-reset randomization).
* :class:`BuildTimeVariationBase` — sampled once and applied to asset configs
  before the env cfg is composed (e.g. picking a dome-light HDR).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, Any

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

if TYPE_CHECKING:
    import torch


@configclass
class VariationBaseCfg:
    """Base configclass for :class:`VariationBase` instances."""

    enabled: bool = False
    """Whether the variation is applied. Opt in via :meth:`VariationBase.enable` or a cfg override."""

    sampler_cfg: SamplerBaseCfg = field(default_factory=SamplerBaseCfg)
    """Declarative sampler driving this variation. Subclasses set a concrete default."""


class VariationBase(ABC):
    """Variation base class.

    This class only enforces that the variation has a name, a config, a sampler,
    a way to enable and disable it, and a way to apply a new config.

    """

    cfg: VariationBaseCfg
    """The configclass instance holding this variation's tunable parameters."""

    name: str
    """Identifier under which this variation is registered on its asset."""

    def __init__(self, cfg: VariationBaseCfg, name: str):
        self.name = name
        self._sampler: SamplerBase | None = None
        self._sample_listeners: list[Callable[[Any, Any], None]] = []
        self.apply_cfg(cfg)

    @property
    def enabled(self) -> bool:
        """Whether this variation is active and should be built into ``events_cfg``."""
        return self.cfg.enabled

    def enable(self) -> None:
        """Mark this variation as active."""
        self.cfg.enabled = True

    def disable(self) -> None:
        """Mark this variation as inactive."""
        self.cfg.enabled = False

    @property
    def sampler(self) -> SamplerBase | None:
        """The sampler driving this variation, or ``None`` if not yet set."""
        return self._sampler

    def add_sample_listener(self, listener: Callable[[Any, torch.Tensor | None], None]) -> None:
        """Subscribe ``listener`` (called as ``listener(sample, env_ids)``) to this variation's samples.

        Listeners are stored on the variation, so ``apply_cfg`` re-binds them onto the
        rebuilt sampler and they survive cfg/sampler swaps.
        """
        self._sample_listeners.append(listener)
        if self._sampler is not None:
            self._sampler.add_listener(listener)

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Apply new ``cfg``.

        Replaces ``cfg`` and rebuilds ``sampler`` from ``cfg.sampler_cfg``, re-binding any
        variation-owned sample listeners onto the new sampler. Subclasses with extra derived
        state should override and call ``super().apply_cfg(cfg)`` first.

        Args:
            cfg: A cfg of the ``VariationBaseCfg`` subclass this variation accepts.
        """
        self.cfg = cfg
        assert isinstance(
            cfg.sampler_cfg, SamplerBaseCfg
        ), f"cfg.sampler_cfg must be a SamplerBaseCfg; got {type(cfg.sampler_cfg).__name__}."
        self._sampler = cfg.sampler_cfg.build()
        # Re-bind variation-owned listeners so a cfg/sampler swap doesn't drop subscriptions.
        for listener in self._sample_listeners:
            self._sampler.add_listener(listener)


class RunTimeVariationBase(VariationBase):
    """Variation realised at run time via an ``EventTermCfg``.

    Use when the underlying property can be flipped during simulation (e.g.
    visual color, initial pose, mass).
    """

    @abstractmethod
    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        """Return the ``(name, cfg)`` event term that realises this variation."""
        ...


class BuildTimeVariationBase(VariationBase):
    """Variation sampled once and applied before the env is built.

    Use for properties that can't change in-flight: HDR maps, USD swaps,
    spawner params baked into a config. Subclasses hold references to the
    asset(s) they mutate.
    """

    @abstractmethod
    def apply(self) -> None:
        """Sample and mutate the bound asset(s) in place to realise this variation.

        Called once per env build, while the variation is enabled.
        """
        ...
