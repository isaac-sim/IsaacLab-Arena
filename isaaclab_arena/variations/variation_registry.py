# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Global registry of :class:`VariationBase` subclasses.

Populated at import time via the :func:`register_variation` decorator.
Not consumed by the builder yet — this establishes the naming contract
used by future CLI resolution (e.g. ``--variation cracker_box.color=...``).
"""

from __future__ import annotations

from typing import ClassVar

from isaaclab_arena.variations.variation_base import VariationBase


class VariationRegistry:
    """Global name → :class:`VariationBase` subclass table.

    Populated at import time via :func:`register_variation`. Duplicate
    registrations raise to catch accidental re-registration.
    """

    _entries: ClassVar[dict[str, type[VariationBase]]] = {}

    @classmethod
    def register(cls, name: str, variation_cls: type[VariationBase]) -> None:
        """Register ``variation_cls`` under ``name``.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in cls._entries:
            raise ValueError(
                f"Variation '{name}' is already registered to "
                f"{cls._entries[name].__module__}.{cls._entries[name].__name__}."
            )
        cls._entries[name] = variation_cls

    @classmethod
    def get(cls, name: str) -> type[VariationBase]:
        """Return the :class:`VariationBase` subclass registered under ``name``."""
        if name not in cls._entries:
            raise KeyError(f"Variation '{name}' is not registered. Known variations: {sorted(cls._entries)}")
        return cls._entries[name]

    @classmethod
    def entries(cls) -> dict[str, type[VariationBase]]:
        """Return a shallow copy of the current registry."""
        return dict(cls._entries)


def register_variation(cls: type[VariationBase]) -> type[VariationBase]:
    """Decorator: register a :class:`VariationBase` subclass under its ``name``.

    The variation's class-level :attr:`~VariationBase.name` attribute is used
    as the registry key — the same name the asset keys it under when it is
    attached via
    :meth:`~isaaclab_arena.assets.object_base.ObjectBase.add_variation`. This
    keeps the "variation name" defined in exactly one place: the class itself.

    Example::

        @register_variation
        class ObjectColorVariation(VariationBase):
            name = "color"
            ...
    """
    assert isinstance(getattr(cls, "name", None), str) and cls.name, (
        f"Variation {cls.__module__}.{cls.__name__} must declare a non-empty "
        "class-level `name: ClassVar[str]` before @register_variation."
    )
    VariationRegistry.register(cls.name, cls)
    return cls
