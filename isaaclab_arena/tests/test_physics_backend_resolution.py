# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for physics backend resolution helpers."""

import pytest

from isaaclab_arena.environments.physics_backend_resolution import assert_same_physics_backend, resolve_physics_backend
from isaaclab_arena.utils.physics_backend import PhysicsBackend


def test_resolve_prefers_cli_presets():
    assert (
        resolve_physics_backend(
            cli_presets=PhysicsBackend.PHYSX,
            env_physics_backend=PhysicsBackend.NEWTON,
        )
        is PhysicsBackend.PHYSX
    )


def test_resolve_uses_env_default_when_cli_unset():
    assert (
        resolve_physics_backend(
            cli_presets=None,
            env_physics_backend=PhysicsBackend.NEWTON,
        )
        is PhysicsBackend.NEWTON
    )
    assert (
        resolve_physics_backend(
            cli_presets=None,
            env_physics_backend=PhysicsBackend.PHYSX,
        )
        is PhysicsBackend.PHYSX
    )


def test_assert_same_physics_backend_passes_for_matching_backends():
    assert_same_physics_backend(
        PhysicsBackend.PHYSX,
        PhysicsBackend.PHYSX,
        context="test",
    )


def test_assert_same_physics_backend_rejects_backend_swap():
    with pytest.raises(AssertionError, match="env_cfg_callback changed the physics backend"):
        assert_same_physics_backend(
            PhysicsBackend.PHYSX,
            PhysicsBackend.NEWTON,
            context="env_cfg_callback",
        )
