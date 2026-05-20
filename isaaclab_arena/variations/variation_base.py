# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base class.

A :class:`VariationBase` describes *one* knob to turn on the scene — a
sampler that drives it and the event term that realises it. Concrete
subclasses are responsible for remembering the target asset (typically by
name, not by reference, to avoid back-edges into the asset graph that can
trip reference-walking validators like
:func:`isaaclab.utils.configclass._validate`). Variations are attached to
their target asset in the asset's ``__init__`` via
:meth:`~isaaclab_arena.assets.object_base.ObjectBase.add_variation`
(disabled by default, pre-configured with a sensible default sampler where
applicable) and then toggled by the user via :meth:`enable` (and optionally
narrowed via :meth:`set_sampler`). The builder walks the scene, collects
enabled variations, and merges their event terms into ``events_cfg``.

The structural knobs of a variation (mode, target sub-mesh, ...) live on a
dedicated :class:`VariationBaseCfg` subclass that each concrete variation
ships alongside its class. The runtime state that is *not* configuration —
the ``enabled`` flag and the current :class:`~isaaclab_arena.variations.sampler.Sampler` —
is managed on the variation object itself (:meth:`enable`, :meth:`set_sampler`).
Keeping them separate avoids the circular-reference trap that
:func:`isaaclab.utils.configclass._validate` imposes on cfg instances
(see ``2026_04_21_variation_system_plan.md``) and leaves the cfg as a
leaf-only, CLI / Hydra-friendly dataclass.
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

    Every concrete variation ships a cfg subclass of this type so downstream
    code (the env builder, the Hydra-driven variation schema in
    :meth:`~isaaclab_arena.environments.arena_env_builder.ArenaEnvBuilder.get_variations_schema`)
    can rely on a single common parent — and on every variation cfg carrying
    an ``enabled`` flag — when collecting per-variation schemas from a scene.

    Attributes:
        enabled: Whether this variation should be applied. Set to ``True`` to
            include the variation in the env's events_cfg (typically via
            :meth:`VariationBase.enable` or a Hydra override
            ``<asset>.<variation>.enabled=true``); defaults to ``False`` so
            assets attach their supported variations in a disabled state and
            users opt in explicitly.

    What does **not** live on the cfg:
        * The target asset — passed explicitly to the variation's ctor and
          stored as a *name* only (never a back-reference, to avoid the
          reference cycles that trip ``configclass._validate``).
        * The live :class:`~isaaclab_arena.variations.sampler.Sampler`
          instance. The cfg stores a :class:`SamplerCfg` (Hydra-friendly,
          plain data); the live sampler is built from it lazily and held on
          the variation object via :meth:`VariationBase.set_sampler`.
    """

    enabled: bool = False


class VariationBase(ABC):
    """Abstract variation.

    A variation binds a target asset and a
    :class:`~isaaclab_arena.variations.sampler.Sampler` to the event term
    required to apply it on reset / prestartup. It starts disabled; concrete
    subclasses typically install a default sampler in their constructor so
    the user can flip it on with a single :meth:`enable` call and override
    the distribution later via :meth:`set_sampler` if desired. Subclasses
    implement :meth:`build_event_cfg`, which the builder calls once per
    enabled variation.

    Concrete subclasses must also declare a class-level :attr:`name` — a short,
    unique identifier used both as the key under which the asset stores the
    variation (see
    :meth:`~isaaclab_arena.assets.object_base.ObjectBase.add_variation`) and as
    the registry key picked up by
    :func:`~isaaclab_arena.variations.variation_registry.register_variation`.

    Concrete subclasses pair themselves with a :class:`VariationBaseCfg`
    subclass that declares their tunable parameters (mode, mesh selector,
    ...) and forward an instance of it to :meth:`__init__`. The base simply
    stores it on :attr:`cfg`; subclasses typically narrow the attribute
    type via a class-level annotation (``cfg: ObjectColorVariationCfg``).
    """

    #: Short, unique identifier for this variation kind (e.g. ``"color"``,
    #: ``"mass"``). Concrete subclasses **must** set this; abstract intermediates
    #: may leave it unset. Used by
    #: :class:`~isaaclab_arena.variations.variation_registry.VariationRegistry`
    #: and :meth:`~isaaclab_arena.assets.object_base.ObjectBase.add_variation`.
    name: ClassVar[str]

    #: The configclass instance holding this variation's tunable parameters.
    #: Subclasses narrow the type via a class-level annotation.
    cfg: VariationBaseCfg

    def __init__(self, cfg: VariationBaseCfg):
        self.cfg = cfg
        self._sampler: Sampler | None = None
        #: Sample observers (typically the variation ledger). Owned by the
        #: variation rather than the sampler so they survive
        #: :meth:`set_sampler` swaps — see :meth:`add_sample_listener`.
        self._sample_listeners: list[Callable[[torch.Tensor], None]] = []

    @property
    def enabled(self) -> bool:
        """Whether this variation is active and should be built into ``events_cfg``.

        Sourced from :attr:`VariationBaseCfg.enabled` so that the cfg is the
        single source of truth — both for the imperative API
        (:meth:`enable` / :meth:`disable`) and for the Hydra-driven path
        (:meth:`~isaaclab_arena.environments.arena_env_builder.ArenaEnvBuilder.apply_hydra_variation_overrides`).
        """
        return self.cfg.enabled

    def enable(self) -> None:
        """Mark this variation as active. A sampler must still be provided via :meth:`set_sampler`."""
        self.cfg.enabled = True

    def disable(self) -> None:
        """Mark this variation as inactive. It will be skipped by the builder."""
        self.cfg.enabled = False

    @property
    def sampler(self) -> Sampler | None:
        """The sampler driving this variation, or ``None`` if not yet set."""
        return self._sampler

    def set_sampler(self, sampler: Sampler | SamplerCfg) -> None:
        """Replace this variation's sampler.

        The argument may be either a declarative :class:`SamplerCfg` (the
        Hydra-/serialisation-friendly form) or a live :class:`Sampler`
        (the imperative form); dispatch is by type:

        * :class:`SamplerCfg` — built into a live sampler via
          :meth:`SamplerCfg.build`, and — if ``self.cfg`` exposes a
          ``sampler`` field (the common case for variations that accept a
          configurable distribution) — also written to
          ``self.cfg.sampler`` so the declarative description stays the
          source of truth and the cfg still survives a round-trip through
          serialisation.
        * :class:`Sampler` — stored directly as the live sampler. This is
          the imperative escape hatch (useful for tests and code-level
          overrides); ``self.cfg`` is **not** touched, so the cfg will no
          longer reflect the live distribution. Callers who want cfg to
          track a live sampler must pass a :class:`SamplerCfg` instead.

        Raises:
            TypeError: If ``sampler`` is neither a :class:`Sampler` nor a
                :class:`SamplerCfg`.
        """
        if isinstance(sampler, SamplerCfg):
            new_sampler = sampler.build()
            new_cfg_sampler: SamplerCfg | None = sampler
        elif isinstance(sampler, Sampler):
            new_sampler = sampler
            new_cfg_sampler = None
        else:
            raise TypeError(f"set_sampler expects a Sampler or SamplerCfg; got {type(sampler).__name__}.")

        # Detach existing listeners from the outgoing sampler so it stops
        # firing into the ledger if anyone else still holds a reference, then
        # re-attach the variation-owned list to the incoming sampler. The
        # variation, not the sampler, is the canonical source of truth for
        # which listeners belong to this variation, so a sampler swap must
        # not silently drop subscriptions.
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

        Listeners are stored on the variation (not the sampler), so a
        subsequent :meth:`set_sampler` call transparently rebinds them to
        the new sampler — i.e. the ledger does not need to know about
        sampler swaps.

        The public entry point for the recording layer
        (:class:`~isaaclab_arena.variations.ledger.VariationLedger`).
        Listeners are called synchronously from inside
        :meth:`Sampler.sample`; observers that need to retain the sample
        across timesteps should detach / clone it themselves.
        """
        self._sample_listeners.append(listener)
        if self._sampler is not None:
            self._sampler.add_listener(listener)

    def remove_sample_listener(self, listener: Callable[[torch.Tensor], None]) -> None:
        """Unsubscribe a previously-registered ``listener``.

        Raises:
            ValueError: If ``listener`` was not registered. Mirrors
                :meth:`Sampler.remove_listener`; bookkeeping errors fail
                loudly rather than silently.
        """
        self._sample_listeners.remove(listener)
        if self._sampler is not None:
            self._sampler.remove_listener(listener)

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Install ``cfg`` as the variation's new source of truth.

        Used by the structured-config / Hydra path
        (:meth:`~isaaclab_arena.environments.arena_env_builder.ArenaEnvBuilder.apply_hydra_variation_overrides`)
        to push a fully-composed cfg back onto a live variation in one shot.
        Replaces :attr:`cfg` wholesale — so every cfg-borne field (``enabled``
        and any variation-specific knobs like ``mode`` / ``mesh_name``) flips
        atomically — and rebuilds the live :class:`Sampler` from the new
        :class:`~isaaclab_arena.variations.sampler.SamplerCfg` if the cfg
        carries one. Subclasses that own additional derived state (something
        cached from the cfg that isn't the sampler) should override this and
        call ``super().apply_cfg(cfg)`` first.

        This is the abstraction boundary that keeps the env builder free of
        variation-specific field names: the builder doesn't enumerate
        ``"sampler"`` / ``"mode"`` / ...; it just hands the composed cfg to
        :meth:`apply_cfg` and lets the variation cfg dataclass *be* the
        enumeration of tunable parameters.

        Args:
            cfg: The cfg to install. Must be an instance of the same
                :class:`VariationBaseCfg` subclass that this variation
                originally accepted (typically constructed by the
                structured-config / Hydra layer from this variation's
                ``*Cfg`` schema). Not type-checked at runtime; assigning
                an incompatible cfg will only fail later, when a
                concrete method tries to read a field that isn't there.
        """
        self.cfg = cfg
        sampler_cfg = getattr(cfg, "sampler", None)
        if isinstance(sampler_cfg, SamplerCfg):
            self.set_sampler(sampler_cfg)

    @abstractmethod
    def build_event_cfg(self, scene: Scene) -> tuple[str, EventTermCfg]:
        """Return the event term that realises this variation.

        Args:
            scene: The arena scene; passed in case a variation needs to
                inspect or resolve other assets (e.g. a scene-wide light
                variation). Simple per-asset variations may ignore it.

        Returns:
            ``(name, cfg)`` pair. ``name`` must be unique across *all*
            enabled variations in the scene — the builder will raise if it
            collides with another event term. Variations that need to fan
            out to multiple assets should be expressed as multiple
            :class:`VariationBase` instances rather than a single variation
            emitting multiple terms.
        """
        ...
