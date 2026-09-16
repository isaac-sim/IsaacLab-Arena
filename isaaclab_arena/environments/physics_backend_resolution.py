# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve and validate the Arena physics backend for environment compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.utils.physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from isaaclab.sim import SimulationCfg


def resolve_physics_backend(
    *,
    cli_presets: PhysicsBackend | None,
    env_physics_backend: PhysicsBackend,
) -> PhysicsBackend:
    """Return the backend selected by CLI presets or the environment default."""
    if cli_presets is not None:
        return cli_presets
    return env_physics_backend


def backend_type_from_sim_cfg(sim_cfg: SimulationCfg) -> PhysicsBackend:
    """Map ``sim_cfg.physics`` to a ``PhysicsBackend`` value."""
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    physics_cfg = sim_cfg.physics
    if physics_cfg is None or isinstance(physics_cfg, PhysxCfg):
        return PhysicsBackend.PHYSX
    assert isinstance(physics_cfg, NewtonCfg), f"Unsupported physics config: {type(physics_cfg).__name__}"
    return PhysicsBackend.NEWTON


def assert_same_physics_backend(
    before: PhysicsBackend,
    after: PhysicsBackend,
    *,
    context: str,
) -> None:
    """Assert a hook did not change the resolved physics backend type."""
    assert before is after, (
        f"{context} changed the physics backend from {before.value!r} to {after.value!r}; "
        "select the backend via --presets or IsaacLabArenaEnvironment.physics_backend instead."
    )
