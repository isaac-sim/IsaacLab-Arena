# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for decorator-based Isaac CAP component registration."""

from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry
from isaaclab_arena_environments.isaac_cap import register_components

_ISAAC_CAP_ENVIRONMENTS = {
    "vabar_contact_rich_insertion_v2__gear_easy",
    "vabar_contact_rich_insertion_v2__gear_easy_pair",
    "vabar_contact_rich_insertion_v2__gear_medium_train",
    "vabar_cable_routing_v2__easy",
    "vabar_cable_routing_v2__medium",
}


def test_isaac_cap_components_registered():
    """Isaac CAP graph names resolve to their decorator-registered components."""
    register_components()

    from isaaclab_arena_environments.isaac_cap.cable_routing_v2.task import CableRoutingTask
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.embodiment import (
        IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
        IndustrialFr3Robotiq2f85Embodiment,
    )
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task import GearMeshTask

    environment_names = set(EnvironmentRegistry().get_all_keys())
    assert _ISAAC_CAP_ENVIRONMENTS <= environment_names

    asset_registry = AssetRegistry()
    assert asset_registry.get_component_by_name("industrial_fr3_robotiq_2f85_v2") is IndustrialFr3Robotiq2f85Embodiment
    assert (
        asset_registry.get_component_by_name("industrial_fr3_robotiq_2f85_differential_ik_v2")
        is IndustrialFr3Robotiq2f85DifferentialIKEmbodiment
    )

    task_registry = TaskRegistry()
    assert task_registry.get_component_by_name("GearMeshTaskV2") is GearMeshTask
    assert task_registry.get_component_by_name("CableRoutingTaskV2") is CableRoutingTask


def test_isaac_cap_registration_is_idempotent():
    """Repeated Isaac CAP discovery leaves every registry unchanged."""
    register_components()
    registries = (AssetRegistry(), TaskRegistry(), EnvironmentRegistry())
    keys_before = tuple(set(registry.get_all_keys()) for registry in registries)

    register_components()

    keys_after = tuple(set(registry.get_all_keys()) for registry in registries)
    assert keys_after == keys_before
