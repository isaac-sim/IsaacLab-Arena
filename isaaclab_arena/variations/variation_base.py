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
from typing import Any, ClassVar

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


@configclass
class VariationBaseCfg:
    """Base configclass for :class:`VariationBase` instances.

    Attributes:
        enabled: Whether the variation is applied. Defaults to ``False`` (opt in
            via :meth:`VariationBase.enable` or a cfg override).
    """

    enabled: bool = False


class VariationBase(ABC):
    """Abstract variation binding an asset, a sampler, and a realisation hook.

    Concrete subclasses declare a class-level :attr:`name`, pair with a
    :class:`VariationBaseCfg` subclass, and inherit one of the flavored bases
    (:class:`RunTimeVariationBase` or :class:`BuildTimeVariationBase`).
    """

    #: Short, unique identifier for this variation kind (e.g. ``"color"``).
    name: ClassVar[str]

    #: The configclass instance holding this variation's tunable parameters.
    cfg: VariationBaseCfg

    def __init__(self, cfg: VariationBaseCfg):
        self.cfg = cfg
        self._sampler: SamplerBase | None = None
        self._sample_listeners: list[Callable[[Any], None]] = []

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

    def set_sampler(self, sampler: SamplerBase | SamplerBaseCfg) -> None:
        """Replace this variation's sampler.

        Accepts a live :class:`SamplerBase` or a :class:`SamplerBaseCfg` (which
        is built into one). Listeners added via :meth:`add_sample_listener`
        survive the swap.
        """
        assert isinstance(
            sampler, (SamplerBase, SamplerBaseCfg)
        ), f"set_sampler expects a SamplerBase or SamplerBaseCfg; got {type(sampler).__name__}."
        if isinstance(sampler, SamplerBaseCfg):
            new_sampler = sampler.build()
            new_cfg_sampler: SamplerBaseCfg | None = sampler
        else:
            new_sampler = sampler
            new_cfg_sampler = None

        # Re-bind variation-owned listeners so a sampler swap doesn't drop subscriptions.
        if self._sampler is not None:
            for listener in self._sample_listeners:
                self._sampler.remove_listener(listener)
        self._sampler = new_sampler
        for listener in self._sample_listeners:
            self._sampler.add_listener(listener)

        if new_cfg_sampler is not None and hasattr(self.cfg, "sampler"):
            self.cfg.sampler = new_cfg_sampler

    def add_sample_listener(self, listener: Callable[[Any], None]) -> None:
        """Subscribe ``listener`` to every sample drawn by this variation's sampler.

        Listeners are stored on the variation, so they survive subsequent
        :meth:`set_sampler` calls.
        """
        self._sample_listeners.append(listener)
        if self._sampler is not None:
            self._sampler.add_listener(listener)

    def remove_sample_listener(self, listener: Callable[[Any], None]) -> None:
        """Unsubscribe a previously-registered ``listener``."""
        self._sample_listeners.remove(listener)
        if self._sampler is not None:
            self._sampler.remove_listener(listener)

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Install ``cfg`` as the variation's new source of truth.

        Replaces :attr:`cfg` and rebuilds the live sampler if the new cfg carries
        one. Subclasses with extra derived state should override and call
        ``super().apply_cfg(cfg)`` first.

        Args:
            cfg: A cfg of the :class:`VariationBaseCfg` subclass this variation
                accepts.
        """
        self.cfg = cfg
        sampler_cfg = getattr(cfg, "sampler", None)
        if isinstance(sampler_cfg, SamplerBaseCfg):
            self.set_sampler(sampler_cfg)


class RunTimeVariationBase(VariationBase):
    """Variation realised at run time via an ``EventTermCfg``.

    Use when the underlying property can be flipped during simulation (e.g.
    visual color, initial pose, mass).
    """

    @abstractmethod
    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        """Return the ``(name, cfg)`` event term that realises this variation.

        The name must be unique across all enabled variations in the scene.
        """
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
