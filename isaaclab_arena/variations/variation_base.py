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
from dataclasses import field
from typing import ClassVar

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


@configclass
class VariationBaseCfg:
    """Base configclass for :class:`VariationBase` instances."""

    enabled: bool = False
    """Whether the variation is applied. Opt in via :meth:`VariationBase.enable` or a cfg override."""

    sampler_cfg: SamplerBaseCfg = field(default_factory=SamplerBaseCfg)
    """Declarative sampler driving this variation. Subclasses set a concrete default."""


class VariationBase(ABC):
    """Abstract variation binding an asset, a sampler, and a realisation hook."""

    #: Short, unique identifier for this variation kind (e.g. ``"color"``).
    name: ClassVar[str]

    #: The configclass instance holding this variation's tunable parameters.
    cfg: VariationBaseCfg

    def __init__(self, cfg: VariationBaseCfg):
        self.cfg = cfg
        self._sampler: SamplerBase | None = None

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
        is built into one).
        """
        assert isinstance(
            sampler, (SamplerBase, SamplerBaseCfg)
        ), f"set_sampler expects a SamplerBase or SamplerBaseCfg; got {type(sampler).__name__}."
        if isinstance(sampler, SamplerBaseCfg):
            self._sampler = sampler.build()
            self.cfg.sampler_cfg = sampler
        else:
            self._sampler = sampler

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Install ``cfg`` as the variation's new source of truth.

        Replaces :attr:`cfg` and rebuilds the live sampler from its sampler cfg.
        Subclasses with extra derived state should override and call
        ``super().apply_cfg(cfg)`` first.

        Args:
            cfg: A cfg of the :class:`VariationBaseCfg` subclass this variation accepts.
        """
        self.cfg = cfg
        self.set_sampler(cfg.sampler_cfg)


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
