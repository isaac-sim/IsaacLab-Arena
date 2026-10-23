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

import hashlib
import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import Any

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


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
        self._sampler: SamplerBase
        self._sample_listeners: list[Callable[[Any, Any], None]] = []
        self._replay_sampler: Callable[[int, torch.Tensor | None], list[Any] | None] | None = None
        self._generator: torch.Generator | None = None
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
    def sampler(self) -> SamplerBase:
        """The sampler driving this variation."""
        return self._sampler

    def add_sample_listener(self, listener: Callable[[Any, torch.Tensor | None], None]) -> None:
        """Subscribe ``listener`` (called as ``listener(sample, env_ids)``) to this variation's samples.

        Listeners are stored on the variation, so ``apply_cfg`` re-binds them onto the
        rebuilt sampler and they survive cfg/sampler swaps.
        """
        self._sample_listeners.append(listener)
        self._sampler.add_listener(listener)

    def set_replay_sampler(
        self,
        replay_sampler: Callable[[int, torch.Tensor | None], list[Any] | None] | None,
    ) -> None:
        """Route future draws through an optional replay sampler."""
        self._replay_sampler = replay_sampler
        self._sampler.set_replay_sampler(replay_sampler)

    def _prepare_at_build_time(self) -> None:
        """Configure prerequisites required before environment construction. Default: no-op.

        A run-time variation overrides this when its later event needs a build-time
        precondition (e.g. forcing its camera untiled so per-env edits take effect).
        """

    def _realize_at_build_time(self) -> None:
        """Sample and mutate the bound asset config(s) in place during construction. Default: no-op.

        A build-time variation realises its whole effect here; a run-time variation leaves it a no-op.
        """

    def configure_at_build_time(self, seed: int, variation_key: str, rebuild_index: int = 0) -> None:
        """Seed this variation's draws and apply its build-time effect for one environment build.

        Args:
            seed: Environment seed.
            variation_key: This variation's ``<asset_name>.<variation_name>`` key.
            rebuild_index: Index of this build among the rebuilds of an Experiment Run, the ``rebuild_index``
                argument of ``ArenaEnvBuilder``.
        """
        # The key gives each variation its own random stream: with the seed alone, equal sampler cfgs draw equal values,
        # and with one shared generator, a variation's values change when others are enabled, disabled or reordered.
        digest = hashlib.sha256(f"{seed}:{variation_key}".encode()).digest()
        # Torch generator seeds are 64-bit, so keep the first 8 bytes of the digest.
        variation_seed = int.from_bytes(digest[:8], "big")
        # Keep the generator on CPU: there, torch.rand and torch.randint give the same values however the draws are
        # split across resets, so the k-th episode to start gets the same run-time values for any num_envs.
        self._generator = torch.Generator(device="cpu").manual_seed(variation_seed)
        self._sampler.set_generator(self._generator)
        self._prepare_at_build_time()
        self._realize_at_build_time()

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Apply new ``cfg``.

        Replaces ``cfg`` and rebuilds ``sampler`` from ``cfg.sampler_cfg``, re-binding the
        variation's generator, replay sampler and sample listeners onto the new sampler. Subclasses
        with extra derived state should override and call ``super().apply_cfg(cfg)`` first.

        Args:
            cfg: A cfg of the ``VariationBaseCfg`` subclass this variation accepts.
        """
        self.cfg = cfg
        assert isinstance(
            cfg.sampler_cfg, SamplerBaseCfg
        ), f"cfg.sampler_cfg must be a SamplerBaseCfg; got {type(cfg.sampler_cfg).__name__}."
        self._sampler = cfg.sampler_cfg.build()
        self._sampler.set_replay_sampler(self._replay_sampler)
        self._sampler.set_generator(self._generator)
        # Re-bind variation-owned listeners so a cfg/sampler swap doesn't drop subscriptions.
        for listener in self._sample_listeners:
            self._sampler.add_listener(listener)


def get_enabled_variations_by_key(variations: dict[str, list[VariationBase]]) -> dict[str, VariationBase]:
    """Return the enabled variations in ``variations`` keyed by ``<asset_name>.<variation_name>``."""
    enabled_variations_by_key: dict[str, VariationBase] = {}
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if not variation.enabled:
                continue
            variation_key = f"{asset_name}.{variation.name}"
            assert variation_key not in enabled_variations_by_key, f"Duplicate enabled variation key: {variation_key!r}"
            enabled_variations_by_key[variation_key] = variation
    return enabled_variations_by_key


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
    asset(s) they mutate and realise the effect in ``_realize_at_build_time``.
    """

    @abstractmethod
    def _realize_at_build_time(self) -> None:
        """Sample and apply this variation to the target configuration once per env build."""
        ...
