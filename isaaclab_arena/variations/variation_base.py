# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base classes.

A :class:`VariationBase` describes one knob to turn on the scene: a sampler
that drives it together with a hook that realises it. Variations are attached
to assets via :meth:`~isaaclab_arena.assets.object_base.ObjectBase.add_variation`,
start disabled, and are flipped on either imperatively (:meth:`VariationBase.enable`)
or via Hydra overrides.

Concrete variations subclass one of two flavors:

* :class:`RunTimeVariationBase` for variations realised via an event term
  that runs during simulation (e.g. per-reset randomization).
* :class:`BuildTimeVariationBase` for variations that sample once and mutate
  asset configs before the env cfg is composed (e.g. picking an HDR for a
  dome light).
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
        enabled: Whether this variation should be applied. Defaults to ``False``
            so users opt in explicitly via :meth:`VariationBase.enable` or a
            Hydra override ``<asset>.<variation>.enabled=true``.
    """

    enabled: bool = False


class VariationBase(ABC):
    """Abstract variation.

    A variation binds a target asset and a
    :class:`~isaaclab_arena.variations.sampler_base.SamplerBase` together with
    a hook that realises the variation. Concrete subclasses declare a
    class-level :attr:`name`, pair themselves with a :class:`VariationBaseCfg`
    subclass, and inherit from one of the two flavored bases
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

        A :class:`SamplerBaseCfg` is built into a live sampler and written back to
        ``self.cfg.sampler`` if the cfg has one (the declarative path). A bare
        :class:`SamplerBase` is stored directly without touching ``self.cfg`` (the
        imperative escape hatch).
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

        Replaces :attr:`cfg` wholesale and rebuilds the live :class:`SamplerBase`
        from the new sampler cfg if the cfg carries one. Subclasses with
        additional derived state should override and call ``super().apply_cfg(cfg)``
        first.

        Args:
            cfg: The cfg to install. Must be an instance of the same
                :class:`VariationBaseCfg` subclass this variation accepts.
        """
        self.cfg = cfg
        sampler_cfg = getattr(cfg, "sampler", None)
        if isinstance(sampler_cfg, SamplerBaseCfg):
            self.set_sampler(sampler_cfg)


class RunTimeVariationBase(VariationBase):
    """Variation that is applied during simulation via an ``EventTermCfg``.

    Concrete subclasses produce an event term that the env's event manager
    invokes at ``reset`` / ``prestartup`` / ``interval``. Use this flavor when
    the underlying property can be flipped at run time (e.g. visual color,
    initial pose, mass).
    """

    @abstractmethod
    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        """Return the event term that realises this variation.

        Returns:
            A ``(name, cfg)`` pair. ``name`` must be unique across all enabled
            variations in the scene.
        """
        ...


class BuildTimeVariationBase(VariationBase):
    """Variation that is sampled once and applied before env build.

    Use this flavor for properties that can't (or shouldn't) be changed
    in-flight during simulation: HDR environment maps, USD asset swaps,
    spawner parameters baked into a config, etc.

    :meth:`apply` is invoked by
    :class:`~isaaclab_arena.environments.arena_env_builder.ArenaEnvBuilder`
    after Hydra overrides have been pushed through :meth:`apply_cfg` and
    before the scene cfg is materialised, so mutations to asset configs are
    visible to env cfg composition. Subclasses are expected to hold direct
    references to the asset(s) they mutate (captured at construction time).
    """

    @abstractmethod
    def apply(self) -> None:
        """Sample and mutate the bound asset(s) in place to realise this variation.

        Called exactly once per env build, while the variation is enabled.
        """
        ...
