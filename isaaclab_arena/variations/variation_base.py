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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import EventTermCfg

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.sampler import Sampler


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
    """

    #: Short, unique identifier for this variation kind (e.g. ``"color"``,
    #: ``"mass"``). Concrete subclasses **must** set this; abstract intermediates
    #: may leave it unset. Used by
    #: :class:`~isaaclab_arena.variations.variation_registry.VariationRegistry`
    #: and :meth:`~isaaclab_arena.assets.object_base.ObjectBase.add_variation`.
    name: ClassVar[str]

    def __init__(self):
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

    def set_sampler(self, sampler: Sampler) -> None:
        """Set the sampler driving this variation."""
        self._sampler = sampler

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
