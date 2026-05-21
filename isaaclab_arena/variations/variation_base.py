# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base class.

A :class:`VariationBase` describes one knob to turn on the scene: a sampler
that drives it and the event term that realises it. Variations are attached
to assets via :meth:`~isaaclab_arena.assets.object_base.ObjectBase.add_variation`,
start disabled, and are flipped on either imperatively (:meth:`VariationBase.enable`)
or via Hydra overrides.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler import Sampler, SamplerCfg

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.scene.scene import Scene


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
    :class:`~isaaclab_arena.variations.sampler.Sampler` to an event term that
    applies it at reset / prestartup. Concrete subclasses declare a class-level
    :attr:`name`, pair themselves with a :class:`VariationBaseCfg` subclass,
    and implement :meth:`build_event_cfg`.
    """

    #: Short, unique identifier for this variation kind (e.g. ``"color"``).
    name: ClassVar[str]

    #: The configclass instance holding this variation's tunable parameters.
    cfg: VariationBaseCfg

    def __init__(self, cfg: VariationBaseCfg):
        self.cfg = cfg
        self._sampler: Sampler | None = None
        self._sample_listeners: list[Callable[[torch.Tensor], None]] = []

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
    def sampler(self) -> Sampler | None:
        """The sampler driving this variation, or ``None`` if not yet set."""
        return self._sampler

    def set_sampler(self, sampler: Sampler | SamplerCfg) -> None:
        """Replace this variation's sampler.

        A :class:`SamplerCfg` is built into a live sampler and written back to
        ``self.cfg.sampler`` if the cfg has one (the declarative path). A bare
        :class:`Sampler` is stored directly without touching ``self.cfg`` (the
        imperative escape hatch).
        """
        assert isinstance(
            sampler, (Sampler, SamplerCfg)
        ), f"set_sampler expects a Sampler or SamplerCfg; got {type(sampler).__name__}."
        if isinstance(sampler, SamplerCfg):
            new_sampler = sampler.build()
            new_cfg_sampler: SamplerCfg | None = sampler
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

    def add_sample_listener(self, listener: Callable[[torch.Tensor], None]) -> None:
        """Subscribe ``listener`` to every sample drawn by this variation's sampler.

        Listeners are stored on the variation, so they survive subsequent
        :meth:`set_sampler` calls.
        """
        self._sample_listeners.append(listener)
        if self._sampler is not None:
            self._sampler.add_listener(listener)

    def remove_sample_listener(self, listener: Callable[[torch.Tensor], None]) -> None:
        """Unsubscribe a previously-registered ``listener``."""
        self._sample_listeners.remove(listener)
        if self._sampler is not None:
            self._sampler.remove_listener(listener)

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Install ``cfg`` as the variation's new source of truth.

        Replaces :attr:`cfg` wholesale and rebuilds the live :class:`Sampler`
        from the new sampler cfg if the cfg carries one. Subclasses with
        additional derived state should override and call ``super().apply_cfg(cfg)``
        first.

        Args:
            cfg: The cfg to install. Must be an instance of the same
                :class:`VariationBaseCfg` subclass this variation accepts.
        """
        self.cfg = cfg
        sampler_cfg = getattr(cfg, "sampler", None)
        if isinstance(sampler_cfg, SamplerCfg):
            self.set_sampler(sampler_cfg)

    @abstractmethod
    def build_event_cfg(self, scene: Scene) -> tuple[str, EventTermCfg]:
        """Return the event term that realises this variation.

        Args:
            scene: The arena scene; passed in case the variation needs to
                resolve other assets.

        Returns:
            A ``(name, cfg)`` pair. ``name`` must be unique across all enabled
            variations in the scene.
        """
        ...
