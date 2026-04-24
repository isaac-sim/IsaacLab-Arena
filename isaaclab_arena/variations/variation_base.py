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
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler import Sampler, SamplerCfg

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene


@configclass
class VariationBaseCfg:
    """Base configclass for :class:`VariationBase` instances.

    Intentionally empty: there are no fields shared across all variation
    kinds today, but every concrete variation ships a cfg subclass of this
    type so downstream code (the env builder, the forthcoming Hydra CLI
    layer in ``hydra_dynamic_schema_example.py``) can rely on a single
    common parent when collecting per-variation schemas from a scene.

    What does **not** live on the cfg:
        * the target asset — passed explicitly to the variation's ctor and
          stored as a *name* only (never a back-reference, to avoid the
          reference cycles that trip ``configclass._validate``);
        * the ``enabled`` flag and the current sampler — these are runtime
          state toggled via :meth:`VariationBase.enable` /
          :meth:`VariationBase.set_sampler`, not configuration.
    """


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
        self._enabled: bool = False
        self._sampler: Sampler | None = None

    @property
    def enabled(self) -> bool:
        """Whether this variation is active and should be built into ``events_cfg``."""
        return self._enabled

    def enable(self) -> None:
        """Mark this variation as active. A sampler must still be provided via :meth:`set_sampler`."""
        self._enabled = True

    def disable(self) -> None:
        """Mark this variation as inactive. It will be skipped by the builder."""
        self._enabled = False

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
            self._sampler = sampler.build()
            if hasattr(self.cfg, "sampler"):
                self.cfg.sampler = sampler
        elif isinstance(sampler, Sampler):
            self._sampler = sampler
        else:
            raise TypeError(f"set_sampler expects a Sampler or SamplerCfg; got {type(sampler).__name__}.")

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
