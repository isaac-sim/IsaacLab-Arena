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
        self.variations: dict[str, VariationBase] = {}

    def add_variation(self, variation: VariationBase) -> None:
        """Attach a variation under its class-level ``name``, replacing any existing one.

        Subclasses call this from their ``__init__`` to declare the variations
        they support.
        """
        self.variations[variation.name] = variation

    def get_variation(self, name: str) -> VariationBase:
        """Return the variation with the given name."""
        assert name in self.variations, (
            f"Asset '{self.name}' ({type(self).__name__}) does not support variation '{name}'. "
            f"Supported variations: {sorted(self.variations)}."
        )
        return self.variations[name]

    def get_variations(self) -> list[VariationBase]:
        """Return every variation attached to this asset, enabled or not."""
        return list(self.variations.values())
