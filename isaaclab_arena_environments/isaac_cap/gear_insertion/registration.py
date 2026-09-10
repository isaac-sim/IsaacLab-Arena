# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Register the self-contained Isaac Cap gear-insertion prototype."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry

from .asset_factories import (
    GEAR_ASSET_ENTRY_POINTS,
    IndustrialEmptyWarehouseDomeLight,
    IndustrialFr3WorkcellTable,
    IndustrialHdrShadowReceiver,
)
from .embodiment import IndustrialFr3Robotiq2f85DifferentialIKEmbodiment, IndustrialFr3Robotiq2f85Embodiment
from .gear_medium_environment import (
    GearInsertionEasyNewtonEnvironment,
    GearInsertionEasyNewtonEnvironmentCfg,
    GearInsertionNewtonEnvironment,
    GearInsertionNewtonEnvironmentCfg,
)
from .task import GearInsertionTask

_registered = False


def _register(registry, component, name: str) -> None:
    """Register one component unless this exact object is already present."""
    if registry.is_registered(name, ensure_loaded=False):
        existing = registry.get_component_by_name(name)
        assert existing is component, f"Conflicting gear-insertion registration for {name!r}."
        return
    registry.register(component, name)


def register_components() -> None:
    """Register Cap assets, embodiment, task, and environment factories."""
    global _registered
    if _registered:
        return

    asset_registry = AssetRegistry()
    for name, factory in GEAR_ASSET_ENTRY_POINTS.items():
        _register(asset_registry, factory, name)
    for asset_class in (
        IndustrialFr3WorkcellTable,
        IndustrialHdrShadowReceiver,
        IndustrialEmptyWarehouseDomeLight,
        IndustrialFr3Robotiq2f85Embodiment,
        IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
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
            assert existing is factory, f"Conflicting gear-insertion environment {factory.name!r}."
            continue
        environment_registry.register_environment(factory, cfg_type)

    _registered = True
