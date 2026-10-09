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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.assets.asset import Asset


@dataclass(frozen=True)
class VariationBuildContext:
    """Resolved environment settings available when configuring a variation."""

    num_envs: int
    """Number of parallel environments in this build."""

    seed: int | None = None
    """Resolved environment seed, when configured."""

    variation_key: str = ""
    """Stable asset and variation name identifying this variation in the build."""

    def __post_init__(self) -> None:
        assert isinstance(self.num_envs, int) and not isinstance(self.num_envs, bool) and self.num_envs > 0


@configclass
class VariationBaseCfg:
    """Base configclass for :class:`VariationBase` instances."""

    enabled: bool = False
    """Whether the variation is applied. Opt in via :meth:`VariationBase.enable` or a cfg override."""

    sample_per_environment: bool = False
    """Whether each environment receives its own sample instead of sharing one sample."""

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

    supported_sample_per_environment: tuple[bool, ...] = (False, True)
    """Sampling scopes implemented by this variation."""

    def __init__(self, cfg: VariationBaseCfg, name: str):
        self.name = name
        self._attached_asset_ref: ReferenceType[Asset] | None = None
        self._sampler: SamplerBase
        self._sample_listeners: list[Callable[[Any, Any], None]] = []
        self._replay_sampler: Callable[[int, torch.Tensor | None], list[Any] | None] | None = None
        self.apply_cfg(cfg)

    @property
    def attached_asset(self) -> Asset | None:
        """Asset that owns this variation, or None before attachment."""
        return self._attached_asset_ref() if self._attached_asset_ref is not None else None

    def attach(self, asset: Asset) -> None:
        """Bind this variation to one asset without changing any explicit target."""
        assert (
            self._attached_asset_ref is None or self.attached_asset is asset
        ), f"Variation '{self.name}' is already attached to another asset."
        self._validate_sampling_scope(self.cfg)
        self._validate_attachment(asset)
        # Environment config validation follows object attributes without tracking cycles.
        self._attached_asset_ref = ref(asset)

    def _validate_attachment(self, asset: Asset) -> None:
        """Check that the proposed host agrees with this variation's explicit target."""

    def _validate_sampling_scope(self, cfg: VariationBaseCfg) -> None:
        assert isinstance(cfg.sample_per_environment, bool), "sample_per_environment must be a boolean."
        assert cfg.sample_per_environment in self.supported_sample_per_environment, (
            f"{type(self).__name__} does not support sample_per_environment={cfg.sample_per_environment}; "
            f"supported values are {self.supported_sample_per_environment}."
        )

    def _validate_configuration(self) -> None:
        """Recheck mutable configuration and the attached target before building."""
        self._validate_sampling_scope(self.cfg)
        asset = self.attached_asset
        if asset is not None:
            self._validate_attachment(asset)

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

    def _prepare_at_build_time(self, context: VariationBuildContext | None = None) -> None:
        """Configure prerequisites required before environment construction. Default: no-op.

        A run-time variation overrides this when its later event needs a build-time
        precondition (e.g. forcing its camera untiled so per-env edits take effect).
        Run-time samplers draw only during simulation; preparation must not draw from them.
        """

    def _realize_at_build_time(self, context: VariationBuildContext | None = None) -> None:
        """Sample and mutate the bound asset config(s) in place during construction. Default: no-op.

        A build-time variation realises its whole effect here; a run-time variation leaves it a no-op.
        """

    def configure_at_build_time(self, context: VariationBuildContext | None = None) -> None:
        """Run this variation's build-time preparation and realization, once per env build."""
        self._validate_configuration()
        self._prepare_at_build_time(context)
        self._realize_at_build_time(context)

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Apply new ``cfg``.

        Replaces ``cfg`` and rebuilds ``sampler`` from ``cfg.sampler_cfg``, re-binding any
        variation-owned sample listeners onto the new sampler. Subclasses with extra derived
        state should override and call ``super().apply_cfg(cfg)`` first.

        Args:
            cfg: A cfg of the ``VariationBaseCfg`` subclass this variation accepts.
        """
        self._validate_sampling_scope(cfg)
        assert isinstance(
            cfg.sampler_cfg, SamplerBaseCfg
        ), f"cfg.sampler_cfg must be a SamplerBaseCfg; got {type(cfg.sampler_cfg).__name__}."
        self.cfg = cfg
        self._sampler = cfg.sampler_cfg.build()
        self._sampler.set_replay_sampler(self._replay_sampler)
        # Re-bind variation-owned listeners so a cfg/sampler swap doesn't drop subscriptions.
        for listener in self._sample_listeners:
            self._sampler.add_listener(listener)


class RunTimeVariationBase(VariationBase):
    """Variation realised at run time via an ``EventTermCfg``.

    Use when the underlying property can be flipped during simulation (e.g.
    visual color, initial pose, mass).
    """

    supported_sample_per_environment = (True,)

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

    supported_sample_per_environment = (False,)

    @abstractmethod
    def _realize_at_build_time(self, context: VariationBuildContext | None = None) -> None:
        """Sample and apply this variation to the target configuration once per env build."""
        ...
