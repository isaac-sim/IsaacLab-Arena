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

import torch
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def step_physics(env: ManagerBasedEnv, num_steps: int, render: bool = False) -> None:
    """Advance physics, optionally rendering each step.

    Args:
        env: The Isaac Lab env to step.
        num_steps: Number of physics steps to advance.
        render: When True, render each step so the settle is visible in the GUI. Defaults to
            False (physics-only) for the rollout path; a no-op visual cost when headless.
    """
    dt = env.unwrapped.sim.get_physics_dt()
    for _ in range(num_steps):
        # Does not perturb metric recoder as no env.step is called.
        env.unwrapped.sim.step(render=render)
        env.unwrapped.scene.update(dt)


def objects_settled_per_episode(
    env: ManagerBasedEnv,
    env_ids: list[int],
    object_names: list[str],
    lin_vel_thresh: float,
    ang_vel_thresh: float,
) -> list[bool]:
    """Settled check for a batch of envs, reading each object's velocity once per env in parallel."""
    if not env_ids:
        return []
    scene = env.unwrapped.scene
    device = env.unwrapped.device
    env_ids_t = torch.as_tensor(env_ids, device=device)
    settled = torch.ones(len(env_ids), dtype=torch.bool, device=device)
    # Note(xinjie.yao): For per-asset loop, no single combined buffer holding each object's velocity.
    # Loop over each asset is unavoidable.
    for name in object_names:
        asset = scene[name]
        lin_velocity = wp.to_torch(asset.data.root_lin_vel_w)[env_ids_t].norm(dim=-1)
        ang_velocity = wp.to_torch(asset.data.root_ang_vel_w)[env_ids_t].norm(dim=-1)
        settled &= (lin_velocity <= lin_vel_thresh) & (ang_velocity <= ang_vel_thresh)
    return settled.tolist()
