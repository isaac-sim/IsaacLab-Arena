# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Discover Arena environment factories published by installed packages."""

from __future__ import annotations

import importlib.metadata
from collections import Counter

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

ENVIRONMENT_ENTRY_POINT_GROUP = "isaaclab_arena.environments"
"""Python entry-point group for externally packaged Arena environment factories."""


def load_external_environment_entry_points() -> None:
    """Load and register environment factories published by installed packages."""
    entry_points = sorted(
        importlib.metadata.entry_points(group=ENVIRONMENT_ENTRY_POINT_GROUP),
        key=lambda entry_point: entry_point.name,
    )
    name_counts = Counter(entry_point.name for entry_point in entry_points)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        raise RuntimeError(
            f"Multiple Arena environment entry points are registered with these names: {duplicate_names}"
        )

    registry = EnvironmentRegistry()
    for entry_point in entry_points:
        factory_type = entry_point.load()
        if not isinstance(factory_type, type) or not issubclass(factory_type, ArenaEnvironmentFactory):
            raise TypeError(
                f"Arena environment entry point {entry_point.name!r} must load an ArenaEnvironmentFactory subclass,"
                f" got {factory_type!r}"
            )
        if factory_type.name != entry_point.name:
            raise ValueError(
                f"Arena environment entry point {entry_point.name!r} loads {factory_type.__name__},"
                f" whose name is {factory_type.name!r}"
            )

        if registry.is_registered(entry_point.name, ensure_loaded=False):
            registered_factory_type = registry.get_component_by_name(entry_point.name)
            if registered_factory_type is not factory_type:
                raise RuntimeError(
                    f"Arena environment {entry_point.name!r} is already registered by"
                    f" {registered_factory_type.__module__}.{registered_factory_type.__qualname__}"
                )
            continue
        register_environment(factory_type)
