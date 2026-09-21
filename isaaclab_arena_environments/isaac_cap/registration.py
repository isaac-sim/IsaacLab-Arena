# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import Isaac CAP component modules so their decorators populate registries."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_asset as _register_asset
from isaaclab_arena.assets.register import register_environment as _register_environment
from isaaclab_arena.assets.register import register_task as _register_task
from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg


def register_asset(component=None, *, name: str | None = None):
    """Register an Isaac CAP asset under its name or an explicit graph name."""
    if name is None:
        return _register_asset if component is None else _register_asset(component)

    def decorator(asset):
        registry = AssetRegistry()
        if registry.is_registered(name, ensure_loaded=False):
            print(f"WARNING: Asset {name} is already registered. Doing nothing.")
        else:
            registry.register(asset, name)
        return asset

    return decorator if component is None else decorator(component)


def register_task(task_type=None, *, name: str | None = None):
    """Register an Isaac CAP task under its class or graph-spec name."""
    if name is None:
        return _register_task if task_type is None else _register_task(task_type)

    def decorator(task):
        registry = TaskRegistry()
        if registry.is_registered(name, ensure_loaded=False):
            print(f"WARNING: Task {name} is already registered. Doing nothing.")
        else:
            registry.register(task, name)
        return task

    return decorator if task_type is None else decorator(task_type)


def register_environment(factory_type=None, *, cfg_type: type[ArenaEnvironmentCfg] | None = None):
    """Register an Isaac CAP environment with an optional inherited config type."""
    if cfg_type is None:
        return _register_environment if factory_type is None else _register_environment(factory_type)

    def decorator(factory):
        registry = EnvironmentRegistry()
        if registry.is_registered(factory.name, ensure_loaded=False):
            print(f"WARNING: Environment {factory.name} is already registered. Doing nothing.")
        else:
            registry.register_environment(factory, cfg_type)
        return factory

    return decorator if factory_type is None else decorator(factory_type)


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
