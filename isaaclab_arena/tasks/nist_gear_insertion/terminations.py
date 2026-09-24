# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for NIST gear insertion tasks."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.nist_gear_insertion.geometry import compute_gear_insertion_success


def gear_mesh_insertion_success(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg,
    fixed_object_cfg: SceneEntityCfg,
    gear_base_offset: tuple[float, float, float],
    gear_peg_height: float,
    success_z_fraction: float,
    xy_threshold: float,
    held_gear_base_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Terminate when the held gear is centered on the peg and lowered to the success depth."""
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
