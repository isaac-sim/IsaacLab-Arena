# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base class.

A :class:`VariationBase` describes *one* knob to turn on the scene — the
target asset (or the scene itself), the sampler that drives it, and the
event term that realises it. The builder collects all enabled variations
and merges their event terms into ``events_cfg``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene


class VariationBase(ABC):
    """Abstract variation.

    A variation binds a target (typically an asset name) and a
    :class:`~isaaclab_arena.variations.sampler.Sampler` to the event term
    required to apply it on reset / prestartup. Subclasses implement
    :meth:`build_event_cfg`, which the builder calls once per enabled
    variation and whose outputs are merged into the environment's
    ``events_cfg``.
    """

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
