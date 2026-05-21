# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Global registry of :class:`VariationBase` subclasses."""

from __future__ import annotations

from typing import ClassVar

from isaaclab_arena.variations.variation_base import VariationBase


class VariationRegistry:
    """Global name → :class:`VariationBase` subclass table.

    Populated at import time via :func:`register_variation`.
    """

    _entries: ClassVar[dict[str, type[VariationBase]]] = {}

    @classmethod
    def register(cls, name: str, variation_cls: type[VariationBase]) -> None:
        """Register ``variation_cls`` under ``name``."""
        assert (
            name not in cls._entries
        ), f"Variation '{name}' is already registered to {cls._entries[name].__module__}.{cls._entries[name].__name__}."
        cls._entries[name] = variation_cls

    @classmethod
    def get(cls, name: str) -> type[VariationBase]:
        """Return the :class:`VariationBase` subclass registered under ``name``."""
        assert name in cls._entries, f"Variation '{name}' is not registered. Known variations: {sorted(cls._entries)}"
        return cls._entries[name]

    @classmethod
    def entries(cls) -> dict[str, type[VariationBase]]:
        """Return a shallow copy of the current registry."""
        return dict(cls._entries)


def register_variation(cls: type[VariationBase]) -> type[VariationBase]:
    """Decorator: register a :class:`VariationBase` subclass under its class-level ``name``."""
    assert isinstance(getattr(cls, "name", None), str) and cls.name, (
        f"Variation {cls.__module__}.{cls.__name__} must declare a non-empty "
        "class-level `name: ClassVar[str]` before @register_variation."
    )
    VariationRegistry.register(cls.name, cls)
    return cls
