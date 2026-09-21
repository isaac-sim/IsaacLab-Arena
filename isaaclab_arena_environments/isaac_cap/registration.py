# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import Isaac CAP component modules so their decorators populate registries."""

from __future__ import annotations

import importlib

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
