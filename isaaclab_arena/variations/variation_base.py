# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base class.

A :class:`VariationBase` describes *one* knob to turn on the scene — the
target asset, the sampler that drives it, and the event term that realises
it. Variations are instantiated by their target asset as part of
:meth:`~isaaclab_arena.assets.object_base.ObjectBase.available_variations`
(disabled + sampler-less by default) and then configured by the user via
:meth:`enable` / :meth:`set_sampler`. The builder walks the scene, collects
enabled variations, and merges their event terms into ``events_cfg``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.sampler import Sampler


class VariationBase(ABC):
    """Abstract variation.

    A variation binds a target asset and a
    :class:`~isaaclab_arena.variations.sampler.Sampler` to the event term
    required to apply it on reset / prestartup. It starts disabled with no
    sampler; the user flips it on via :meth:`enable` and supplies a sampler
    via :meth:`set_sampler` before the environment is built. Subclasses
    implement :meth:`build_event_cfg`, which the builder calls once per
    enabled variation.
    """

    def __init__(self, asset: ObjectBase):
        self.asset = asset
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
