# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Register assets, embodiments, tasks, and environments provided by Isaac Cap."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry

_registered = False
# Task-package imports call back into this shared entry point.
_registering = False


def _register(registry, component, name: str) -> None:
    """Register one component unless this exact object is already present."""
    if registry.is_registered(name, ensure_loaded=False):
        existing = registry.get_component_by_name(name)
        assert existing is component, f"Conflicting Isaac Cap registration for {name!r}."
        return
    registry.register(component, name)


def register_components() -> None:
    """Register every Isaac Cap component through one shared entry point."""
    global _registered, _registering
    if _registered or _registering:
        return

    _registering = True
    try:
        asset_registry = AssetRegistry()
        _register_insertion_task_embodiments(asset_registry)
        _register_gear_insertion_components(asset_registry)
        _registered = True
    finally:
        _registering = False


def _register_insertion_task_embodiments(asset_registry: AssetRegistry) -> None:
    """Register embodiments shared by the insertion-task environments."""
    from .embodiments.insertion_task import (
        IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
        IndustrialFr3Robotiq2f85Embodiment,
    )

    for embodiment_class in (
        IndustrialFr3Robotiq2f85Embodiment,
        IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
    ):
        _register(asset_registry, embodiment_class, embodiment_class.name)


def _register_gear_insertion_components(asset_registry: AssetRegistry) -> None:
    """Register the gear-insertion components."""
    from .gear_insertion.asset_factories import (
        GEAR_ASSET_ENTRY_POINTS,
        IndustrialEmptyWarehouseDomeLight,
        IndustrialFr3WorkcellTable,
        IndustrialHdrShadowReceiver,
    )
    from .gear_insertion.gear_medium_environment import (
        GearInsertionEasyNewtonEnvironment,
        GearInsertionEasyNewtonEnvironmentCfg,
        GearInsertionNewtonEnvironment,
        GearInsertionNewtonEnvironmentCfg,
    )
    from .gear_insertion.task import GearInsertionTask

    for name, factory in GEAR_ASSET_ENTRY_POINTS.items():
        _register(asset_registry, factory, name)
    for asset_class in (
        IndustrialFr3WorkcellTable,
        IndustrialHdrShadowReceiver,
        IndustrialEmptyWarehouseDomeLight,
    ):
        _register(asset_registry, asset_class, asset_class.name)

    _register(TaskRegistry(), GearInsertionTask, GearInsertionTask.__name__)

    environment_registry = EnvironmentRegistry()
    for factory, cfg_type in (
        (GearInsertionNewtonEnvironment, GearInsertionNewtonEnvironmentCfg),
        (GearInsertionEasyNewtonEnvironment, GearInsertionEasyNewtonEnvironmentCfg),
    ):
        if environment_registry.is_registered(factory.name, ensure_loaded=False):
            existing = environment_registry.get_component_by_name(factory.name)
            assert existing is factory, f"Conflicting Isaac Cap environment {factory.name!r}."
            continue
        environment_registry.register_environment(factory, cfg_type)
