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


def _register_environment(factory, cfg_type) -> None:
    """Register one environment unless this exact factory is already present."""
    registry = EnvironmentRegistry()
    if registry.is_registered(factory.name, ensure_loaded=False):
        existing = registry.get_component_by_name(factory.name)
        assert existing is factory, f"Conflicting Isaac Cap environment {factory.name!r}."
        return
    registry.register_environment(factory, cfg_type)


def register_components() -> None:
    """Register every Isaac Cap component through one shared entry point."""
    global _registered, _registering
    if _registered or _registering:
        return

    _registering = True
    try:
        asset_registry = AssetRegistry()
        _register_insertion_task_embodiments(asset_registry)
        _register_cable_routing_embodiment(asset_registry)
        _register_gear_insertion_components(asset_registry)
        _register_cable_routing_components()
        _register_syringe_sort_components(asset_registry)
        _register_usbc_insertion_components(asset_registry)
        _register_tool_sorting_components(asset_registry)
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


def _register_cable_routing_embodiment(asset_registry: AssetRegistry) -> None:
    """Register the bimanual YAM embodiment used for cable routing."""
    from .embodiments.cable_routing import IndustrialBimanualYamEmbodiment

    _register(asset_registry, IndustrialBimanualYamEmbodiment, IndustrialBimanualYamEmbodiment.name)


def _register_cable_routing_components() -> None:
    """Register the cable-routing task and environments."""
    from .cable_routing.environment import (
        CableRoutingEasyEnvironment,
        CableRoutingEasyEnvironmentCfg,
        CableRoutingMediumEnvironment,
        CableRoutingMediumEnvironmentCfg,
    )
    from .cable_routing.task import CableRoutingTask

    _register(TaskRegistry(), CableRoutingTask, CableRoutingTask.__name__)

    for factory, cfg_type in (
        (CableRoutingMediumEnvironment, CableRoutingMediumEnvironmentCfg),
        (CableRoutingEasyEnvironment, CableRoutingEasyEnvironmentCfg),
    ):
        _register_environment(factory, cfg_type)


def _register_usbc_insertion_components(asset_registry: AssetRegistry) -> None:
    """Register the USB-C assets, shared task, and both environments."""
    from .usbc_insertion.assets import USBC_ASSET_CLASSES
    from .usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )
    from .usbc_insertion.task import UsbcInsertionTask

    for asset_class in USBC_ASSET_CLASSES:
        _register(asset_registry, asset_class, asset_class.name)

    _register(TaskRegistry(), UsbcInsertionTask, UsbcInsertionTask.__name__)

    for factory, cfg_type in (
        (UsbcInsertionEasyEnvironment, UsbcInsertionEasyEnvironmentCfg),
        (UsbcInsertionMediumEnvironment, UsbcInsertionMediumEnvironmentCfg),
    ):
        _register_environment(factory, cfg_type)


def _register_gear_insertion_components(asset_registry: AssetRegistry) -> None:
    """Register the gear-insertion components."""
    from .gear_insertion.asset_factories import (
        GEAR_ASSET_ENTRY_POINTS,
        IndustrialEmptyWarehouseDomeLight,
        IndustrialFr3WorkcellTable,
        IndustrialHdrShadowReceiver,
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


def _register_syringe_sort_components(asset_registry: AssetRegistry) -> None:
    """Register the syringe assets, success task, and environment."""
    from . import cap_policy  # noqa: F401
    from .syringe_sort.environments.assets import InstrumentTray, SharpsContainer, SyringeRedCap, SyringeWhiteCap
    from .syringe_sort.environments.environment import (
        SyringeBothEnvironment,
        SyringeBothEnvironmentCfg,
        SyringeClutteredEnvironment,
        SyringeClutteredEnvironmentCfg,
        SyringeSingleEnvironment,
        SyringeSortEnvironmentCfg,
    )
    from .syringe_sort.tasks.task import SyringeSortTask

    for asset_class in (SyringeRedCap, SyringeWhiteCap, InstrumentTray, SharpsContainer):
        _register(asset_registry, asset_class, asset_class.name)
    _register(TaskRegistry(), SyringeSortTask, SyringeSortTask.__name__)
    for factory, cfg in (
        (SyringeSingleEnvironment, SyringeSortEnvironmentCfg),
        (SyringeBothEnvironment, SyringeBothEnvironmentCfg),
        (SyringeClutteredEnvironment, SyringeClutteredEnvironmentCfg),
    ):
        _register_environment(factory, cfg)


def _register_tool_sorting_components(asset_registry: AssetRegistry) -> None:
    """Register easy tool-sort assets, embodiment, and task."""
    from .tool_sorting.assets import TOOL_SORT_ASSET_CLASSES
    from .tool_sorting.embodiment import ToolSortingFr3Robotiq2f85Embodiment
    from .tool_sorting.task import ObjectsInRegionsTask

    _register(asset_registry, ToolSortingFr3Robotiq2f85Embodiment, ToolSortingFr3Robotiq2f85Embodiment.name)
    for asset_class in TOOL_SORT_ASSET_CLASSES:
        _register(asset_registry, asset_class, asset_class.name)
    _register(TaskRegistry(), ObjectsInRegionsTask, ObjectsInRegionsTask.__name__)
