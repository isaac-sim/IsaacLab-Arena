# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab_arena.relations.initializers.anchor_initializer import AnchorInitializer
from isaaclab_arena.relations.initializers.placement_initializer_base import InitializerType, PlacementInitializerBase


def create_initializer(initializer_type: InitializerType) -> PlacementInitializerBase:
    """Return a new initializer of the requested type."""
    initializers_by_type: dict[InitializerType, type[PlacementInitializerBase]] = {
        InitializerType.ANCHOR: AnchorInitializer,
    }
    assert initializer_type in initializers_by_type, f"No initializer registered for {initializer_type}."
    return initializers_by_type[initializer_type]()
