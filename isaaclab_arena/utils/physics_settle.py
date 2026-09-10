# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.tasks.predicates.object_settling import compute_objects_settled_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def step_physics(env: ManagerBasedEnv, num_steps: int, render: bool = False) -> None:
    """Advance physics, optionally rendering each step.

    Args:
        env: The Isaac Lab env to step.
        num_steps: Number of physics steps to advance.
        render: When True, render each step so the settle is visible in the GUI. Defaults to
            False (physics-only).
    """
    dt = env.unwrapped.sim.get_physics_dt()
    for _ in range(num_steps):
        # Does not perturb metric recorder as no env.step is called.
        env.unwrapped.sim.step(render=render)
        env.unwrapped.scene.update(dt)


def are_all_objects_settled_per_env(
    env: ManagerBasedEnv,
    env_ids: list[int],
    object_names: list[str],
    lin_vel_thresh: float,
    ang_vel_thresh: float,
) -> list[bool]:
    """Settled check for a batch of envs, reading each object's velocity once per env in parallel."""
    if not env_ids:
        return []
    arena_env = env.unwrapped
    settled_mask = compute_objects_settled_mask(
        arena_env.arena_world,
        arena_env.scene,
        object_names,
        lin_vel_thresh,
        ang_vel_thresh,
    )
    environment_ids = torch.as_tensor(env_ids, device=arena_env.device)
    return settled_mask[environment_ids].tolist()
