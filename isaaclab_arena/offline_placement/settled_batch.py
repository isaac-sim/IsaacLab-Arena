# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sample pooled placements and capture their state before and after physics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_result import PlacementResult


@dataclass
class SettledBatch:
    """One sampled batch with N environments and B links per articulation.

    Captured tensors own their storage and remain valid after further simulation.
    Frames E, W, R, and L denote the environment, world, measured root, and articulation link.
    """

    source_layouts: dict[int, PlacementResult]
    """Copied source poses and solver checklists by absolute environment ID; asset identities are shared."""
    env_ids: list[int]
    """Absolute environment IDs whose source layouts passed required solver checks."""
    initial_root_poses: dict[str, torch.Tensor]
    """Initial T_E_R root poses (N, 7), xyz metres and xyzw rotation, by scene key."""
    final_root_poses: dict[str, torch.Tensor]
    """Final T_E_R root poses (N, 7), xyz metres and xyzw rotation, by scene key."""
    initial_link_poses: dict[str, torch.Tensor]
    """Initial T_R_L link poses (N, B, 7), xyz metres and xyzw rotation, by articulation key."""
    final_link_poses: dict[str, torch.Tensor]
    """Final T_R_L link poses (N, B, 7), xyz metres and xyzw rotation, by articulation key."""
    final_root_velocities: dict[str, torch.Tensor]
    """Final world-frame root velocities (N, 6), linear xyz m/s then angular xyz rad/s, by scene key."""


def sample_and_settle_batch(
    env: ManagerBasedEnv,
    *,
    root_keys: Sequence[str],
    link_keys: Sequence[str],
    num_env_steps: int,
    render: bool = False,
    log_progress: bool = False,
) -> SettledBatch:
    """Consume one pooled reset and capture poses and final velocities around physics.

    Physics advances only when at least one source candidate passes its required
    solver checks. All candidates remain in source_layouts, including failures.
    The caller owns env, which remains open at its final state.

    Args:
        env: Built environment with a pooled placement reset event.
        root_keys: Rigid-object and articulation scene keys whose roots are measured.
        link_keys: Articulation scene keys whose root-relative links are measured.
        num_env_steps: Positive number of environment steps, each containing decimation substeps.
        render: Render each physics substep.
        log_progress: Print physics-step progress.

    Returns:
        Source candidates and captured measurements for every environment.
    """
    import torch

    from isaaclab_arena.offline_placement.pool_validation import solver_validation_failure, step_placement_physics
    from isaaclab_arena.relations.placement_events import get_placement_pool, get_reset_placement_results

    env = env.unwrapped
    assert num_env_steps > 0, "num_env_steps must be positive"
    assert root_keys, "Sampling requires at least one rigid-object or articulation root"
    placement_pool = get_placement_pool(env)
    assert placement_pool is not None, "Sampling requires a pooled placement reset event"
    assert placement_pool.num_envs == env.num_envs, "Placement pool and scene must have the same environment count"
    env.reset()
    source_layouts = {}
    for env_id, layout in get_reset_placement_results(env).items():
        checklist = layout.validation_results
        source_layouts[env_id] = replace(
            layout,
            positions=dict(layout.positions),
            orientations=dict(layout.orientations),
            validation_results=replace(
                checklist,
                validation_results=dict(checklist.validation_results),
                required_checks=None if checklist.required_checks is None else set(checklist.required_checks),
            ),
        )
    assert set(source_layouts) == set(range(env.num_envs)), "Reset must place every environment"
    env_ids = [env_id for env_id, layout in source_layouts.items() if solver_validation_failure(layout) is None]
    initial_root_poses = {key: env.arena_world.get_pose_e(key) for key in root_keys}
    initial_link_poses = capture_articulation_link_poses_in_root_frame(env, link_keys)
    if env_ids:
        step_placement_physics(env, num_env_steps, render=render, log_progress=log_progress)
    final_root_poses = {key: env.arena_world.get_pose_e(key) for key in root_keys}
    final_link_poses = capture_articulation_link_poses_in_root_frame(env, link_keys)
    final_root_velocities = {}
    for key in root_keys:
        linear_velocity_w = env.arena_world.get_root_linear_velocity_w(key)
        angular_velocity_w = env.arena_world.get_root_angular_velocity_w(key)
        final_root_velocities[key] = torch.cat((linear_velocity_w, angular_velocity_w), dim=-1)
    return SettledBatch(
        source_layouts=source_layouts,
        env_ids=env_ids,
        initial_root_poses=initial_root_poses,
        final_root_poses=final_root_poses,
        initial_link_poses=initial_link_poses,
        final_link_poses=final_link_poses,
        final_root_velocities=final_root_velocities,
    )


def capture_articulation_link_poses_in_root_frame(
    env: ManagerBasedEnv, articulation_keys: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Capture link-to-root poses for N environments and B links per articulation.

    Root-relative poses isolate internal articulation motion from motion of the whole root.
    Comparing them before and after physics detects link drift independently of root drift.
    Frames W, R, and L denote the world, articulation root, and link.

    Args:
        env: Built environment containing the selected articulations.
        articulation_keys: Articulation scene keys to measure.

    Returns:
        Owned T_R_L pose tensors (N, B, 7), with xyz metres and xyzw rotation, by scene key.
    """
    import torch

    from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

    env = env.unwrapped
    poses = {}
    for key in articulation_keys:
        body = env.scene.articulations[key]
        T_W_L = body.data.body_link_pose_w.torch
        T_W_R = env.arena_world.get_pose_w(key)[:, None, :].expand_as(T_W_L)
        t_R_L = quat_apply_inverse(T_W_R[..., 3:], T_W_L[..., :3] - T_W_R[..., :3])
        q_R_L = quat_mul(quat_conjugate(T_W_R[..., 3:]), T_W_L[..., 3:])
        poses[key] = torch.cat((t_R_L, q_R_L), dim=-1)
    return poses
