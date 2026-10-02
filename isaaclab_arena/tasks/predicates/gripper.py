# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless predicates for grippers."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.gripper import Gripper
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def gripper_released(
    env: IsaacLabArenaManagerBasedRLEnv,
    gripper: Gripper,
    grasp_width_m: float,
    release_clearance_m: float = 1.5e-3,
) -> torch.Tensor:
    """Check that the measured gripper opening clears the object's grasp width.

    This checks current clearance, not whether a grasp happened earlier. An open
    gripper can pass before grasping; an opening command alone cannot make it pass.

    Args:
        env: Environment supplying measured joint positions through ArenaWorld.
        gripper: Embodiment-owned width-reporting gripper implementation.
        grasp_width_m: Object width at the grasp, in meters.
        release_clearance_m: Required extra gap beyond the object width, in meters.
            The comparison is strict, so exactly this clearance does not pass.

    Returns:
        Boolean tensor with one result per environment.
    """
    assert math.isfinite(grasp_width_m) and grasp_width_m > 0.0, "Grasp width must be positive and finite."
    assert (
        math.isfinite(release_clearance_m) and release_clearance_m >= 0.0
    ), "Release clearance must be non-negative and finite."
    opening_width_m = gripper.get_opening_width_m(env.arena_world)
    return opening_width_m > grasp_width_m + release_clearance_m


def gripper_not_grasping(
    env: IsaacLabArenaManagerBasedRLEnv,
    gripper: Gripper,
    grasp_width_m: float,
    gap_band_m: float,
    stall_margin_m: float,
) -> torch.Tensor:
    """Check that the hand is not stalled closing around the object's grasp width.

    A closed empty hand or an opening command can pass this check.

    Args:
        env: Environment supplying the gripper's measured and commanded state.
        gripper: Embodiment-owned opening and closing-error measurements.
        grasp_width_m: Object width at the grasp, in meters.
        gap_band_m: Strict tolerance around the grasp width, in meters.
        stall_margin_m: Strict minimum closing displacement error, in meters.

    Returns:
        True for each environment where the hand does not meet both grasp conditions.
    """
    assert math.isfinite(grasp_width_m) and grasp_width_m > 0.0, "grasp_width_m must be positive and finite."
    assert math.isfinite(gap_band_m) and gap_band_m > 0.0, "gap_band_m must be positive and finite."
    assert math.isfinite(stall_margin_m) and stall_margin_m >= 0.0, "stall_margin_m must be non-negative and finite."
    gap = gripper.get_opening_width_m(env.arena_world)
    closing_error = gripper.get_closing_error_m(env)
    gripped = (closing_error > stall_margin_m) & (torch.abs(gap - grasp_width_m) < gap_band_m)
    return ~gripped
