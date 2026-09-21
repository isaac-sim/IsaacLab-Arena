# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that all environments in isaaclab_arena_environments are registered."""

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena_environments.cli import ensure_environments_registered


def test_environments_registered():
    """All environments in isaaclab_arena_environments are in the registry after ensure_environments_registered()."""
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    registered = set(env_registry.get_all_keys())
    assert len(registered) > 0, "No environments registered"


def test_environments_registered_twice():
    """Calling ensure_environments_registered() twice should leave the registry unchanged."""
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    keys_after_first = set(env_registry.get_all_keys())

    ensure_environments_registered()
    keys_after_second = set(env_registry.get_all_keys())

    assert keys_after_first == keys_after_second
