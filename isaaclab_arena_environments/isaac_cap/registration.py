# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import Isaac CAP component modules so their decorators populate registries."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment as _register_environment
from isaaclab_arena.assets.registries import EnvironmentRegistry

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg


def register_environment(factory_type=None, *, cfg_type: type[ArenaEnvironmentCfg] | None = None):
    """Register an Isaac CAP environment with an optional inherited config type."""

    def decorator(factory):
        registry = EnvironmentRegistry()
        if registry.is_registered(factory.name, ensure_loaded=False):
            _assert_same_component(registry, factory, factory.name, "environment")
        elif cfg_type is None:
            _register_environment(factory)
        else:
            registry.register_environment(factory, cfg_type)
        return factory

    return decorator if factory_type is None else decorator(factory_type)


def _assert_same_component(registry, component, name: str, kind: str) -> None:
    """Reject a registry key already owned by a different component."""
    existing = registry.get_component_by_name(name)
    assert existing is component, f"Conflicting Isaac CAP {kind} registration for {name!r}."


_registered = False
_registering = False

# Modules containing every decorator-registered Isaac CAP component.
_COMPONENT_MODULES = (
    ".cap_policy",
    ".embodiments.insertion_task",
    ".embodiments.cable_routing",
    ".gear_insertion.asset_factories",
    ".gear_insertion.task.task",
    ".cable_routing.task",
    ".cable_routing.environment",
    ".syringe_sort.environments.assets",
    ".syringe_sort.tasks.task",
    ".syringe_sort.environments.environment",
    ".usbc_insertion.assets",
    ".usbc_insertion.task",
    ".usbc_insertion.environment",
    ".tool_sorting.assets",
    ".tool_sorting.embodiment",
    ".tool_sorting.task",
    ".gear_insertion_v2.asset_factories",
    ".gear_insertion_v2.embodiment",
    ".gear_insertion_v2.task.task",
    ".gear_insertion_v2.gear_mesh_environment",
    ".cable_routing_v2.task",
    ".cable_routing_v2.environment",
)


def register_components() -> None:
    """Import every Isaac CAP component module exactly once."""
    global _registered, _registering
    if _registered or _registering:
        return

    _registering = True
    try:
        for module_name in _COMPONENT_MODULES:
            importlib.import_module(module_name, package=__package__)
        _registered = True
    finally:
        _registering = False
