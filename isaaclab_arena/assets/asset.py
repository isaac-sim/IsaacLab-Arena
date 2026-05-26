# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase


class Asset:
    """
    Base class for all assets.
    """

    def __init__(self, name: str, tags: list[str] | None = None, **kwargs):
        # NOTE: Cooperative Multiple Inheritance Pattern.
        # Calling super even though this is a base class to support
        # multiple inheritance of inheriting classes.
        super().__init__(**kwargs)
        assert name is not None, "Name is required for all assets"
        self.name = name
        self.tags = tags

    def add_variation(self, variation: VariationBase) -> None:
        """Attach a variation to this asset under its class-level ``name``.

        Subclasses call this from their ``__init__`` to declare the variations
        they support. Re-registering the same name overwrites the existing entry.
        The ``_variations`` dict is created on first attach so subclasses that
        don't route through ``Asset.__init__`` (e.g. some
        :class:`~isaaclab_arena.embodiments.embodiment_base.EmbodimentBase`
        subclasses) still work.
        """
        if not hasattr(self, "_variations"):
            self._variations: dict[str, VariationBase] = {}
        self._variations[variation.name] = variation

    def get_variation(self, name: str) -> VariationBase:
        """Return the variation with the given name."""
        variations = getattr(self, "_variations", {})
        assert name in variations, (
            f"Asset '{self.name}' ({type(self).__name__}) does not support variation '{name}'. "
            f"Supported variations: {sorted(variations)}."
        )
        return variations[name]

    def get_variations(self) -> list[VariationBase]:
        """Return every variation attached to this asset, enabled or not."""
        return list(getattr(self, "_variations", {}).values())
