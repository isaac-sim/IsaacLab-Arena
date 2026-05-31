# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for gear insertion tasks."""

from __future__ import annotations

import torch

import warp as wp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.nist_gear_insertion.geometry import compute_gear_insertion_success


def gear_mesh_insertion_success(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg,
    fixed_object_cfg: SceneEntityCfg,
    gear_base_offset: tuple[float, ...],
    held_gear_base_offset: list[float] | None = None,
    gear_peg_height: float = 0.02,
    success_z_fraction: float = 0.30,
    xy_threshold: float = 0.0025,
    disable_success_termination: bool = False,
) -> torch.Tensor:
    """Terminate when the held gear is centered on the peg and lowered to the success depth.

    When success termination is disabled, the same term remains registered but
    always returns false. This keeps success-rate metrics available during RL
    training while allowing policies to continue after first success.
    """
    if disable_success_termination:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return compute_gear_insertion_success(
        env=env,
        gear_cfg=held_object_cfg,
        board_cfg=fixed_object_cfg,
        peg_offset=gear_base_offset,
        held_gear_base_offset=held_gear_base_offset if held_gear_base_offset is not None else gear_base_offset,
        gear_peg_height=gear_peg_height,
        z_fraction=success_z_fraction,
        xy_threshold=xy_threshold,
    )


def gear_dropped_from_gripper(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "panda_hand",
    distance_threshold: float = 0.15,
) -> torch.Tensor:
    """Reset when the gear has fallen too far from the end-effector.

    This optional guard is useful for evaluation and non-training rollouts. It
    is disabled during recovery-focused training so the policy can learn from
    off-nominal states.
    """
    gear: RigidObject = env.scene[gear_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    eef_indices, _ = robot.find_bodies([ee_body_name])
    ee_pos = wp.to_torch(robot.data.body_pos_w)[:, eef_indices[0]]
    gear_pos = wp.to_torch(gear.data.root_pos_w)
    distance = torch.norm(gear_pos - ee_pos, dim=-1)
    return distance > distance_threshold
