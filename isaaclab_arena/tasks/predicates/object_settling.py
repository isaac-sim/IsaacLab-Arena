# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Instantaneous object velocity checks without recording side effects."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.utils.physics_settle import (
    DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
    DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    compute_objects_settled_mask,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def objects_below_velocity_thresholds(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_names: list[str],
    lin_vel_threshold: float = DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    ang_vel_threshold: float = DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
) -> torch.Tensor:
    """Return whether all named objects meet their speed limits in each environment.

    This check does not record resting positions or count consecutive steps.
    """

    return compute_objects_settled_mask(
        env.arena_world,
        env.scene,
        object_names,
        lin_vel_threshold,
        ang_vel_threshold,
    )
