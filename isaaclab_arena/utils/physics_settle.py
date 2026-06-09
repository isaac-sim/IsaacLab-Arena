# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-side primitives for the post-reset physics settle pass.

These functions are the only part of the settle check that touches the live SimulationApp:
advancing physics and reading back object velocities. They are intentionally decoupled from the
placement pool and its config so the high-level re-selection loop can swap in any other
sim-app-bringup routine without changing its interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def step_physics(env: ManagerBasedEnv, num_steps: int) -> None:
    """Advance physics only without running rendering."""
    dt = env.unwrapped.sim.get_physics_dt()
    for _ in range(num_steps):
        env.unwrapped.sim.step(render=False)
        env.unwrapped.scene.update(dt)


def max_object_velocities(env: ManagerBasedEnv, env_id: int, object_names: list[str]) -> list[tuple[float, float]]:
    """Read back each object's (linear, angular) world-frame velocity for one env, in object_names order."""
    scene = env.unwrapped.scene
    velocities: list[tuple[float, float]] = []
    for name in object_names:
        asset = scene[name]
        lin_velocity = wp.to_torch(asset.data.root_lin_vel_w)[env_id].norm().item()
        ang_velocity = wp.to_torch(asset.data.root_ang_vel_w)[env_id].norm().item()
        velocities.append((lin_velocity, ang_velocity))
    return velocities


def objects_settled(
    env: ManagerBasedEnv,
    env_id: int,
    object_names: list[str],
    lin_vel_thresh: float,
    ang_vel_thresh: float,
) -> bool:
    """True when every named object in this env is below both the linear and angular velocity thresholds."""
    return all(
        lin_velocity <= lin_vel_thresh and ang_velocity <= ang_vel_thresh
        for lin_velocity, ang_velocity in max_object_velocities(env, env_id, object_names)
    )
