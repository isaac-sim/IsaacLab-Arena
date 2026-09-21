# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that all environments in isaaclab_arena_environments are registered."""

from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry
from isaaclab_arena_environments.cli import ensure_environments_registered

_ISAAC_CAP_ENVIRONMENTS = {
    "vabar_contact_rich_insertion_v2__gear_easy",
    "vabar_contact_rich_insertion_v2__gear_easy_pair",
    "vabar_contact_rich_insertion_v2__gear_medium_train",
    "vabar_cable_routing_v2__easy",
    "vabar_cable_routing_v2__medium",
}


def test_environments_registered():
    """All environments in isaaclab_arena_environments are in the registry after ensure_environments_registered()."""
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    registered = set(env_registry.get_all_keys())
    assert len(registered) > 0, "No environments registered"
    assert _ISAAC_CAP_ENVIRONMENTS <= registered


def test_isaac_cap_compatibility_names_registered():
    """Isaac CAP graph names resolve to their decorator-registered components."""
    ensure_environments_registered()

    from isaaclab_arena_environments.isaac_cap.cable_routing_v2.task import CableRoutingTask
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.embodiment import (
        IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
        IndustrialFr3Robotiq2f85Embodiment,
    )
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task import GearMeshTask

    asset_registry = AssetRegistry()
    assert asset_registry.get_component_by_name("industrial_fr3_robotiq_2f85_v2") is IndustrialFr3Robotiq2f85Embodiment
    assert (
        asset_registry.get_component_by_name("industrial_fr3_robotiq_2f85_differential_ik_v2")
        is IndustrialFr3Robotiq2f85DifferentialIKEmbodiment
    )

    task_registry = TaskRegistry()
    assert task_registry.get_component_by_name("GearMeshTaskV2") is GearMeshTask
    assert task_registry.get_component_by_name("CableRoutingTaskV2") is CableRoutingTask


def test_environments_registered_twice():
    """Calling ensure_environments_registered() twice should leave the registry unchanged."""
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    keys_after_first = set(env_registry.get_all_keys())

    ensure_environments_registered()
    keys_after_second = set(env_registry.get_all_keys())

    assert keys_after_first == keys_after_second
