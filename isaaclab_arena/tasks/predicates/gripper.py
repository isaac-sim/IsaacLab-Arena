# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless predicates for parallel-jaw grippers."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def parallel_jaw_gripper_released(
    env: IsaacLabArenaManagerBasedRLEnv,
    robot_name: str,
    gripper_joint_name: str,
    jaw_gap_at_zero_joint_m: float,
    grasp_width_m: float,
    release_clearance_m: float = 1.5e-3,
) -> torch.Tensor:
    """Check that the measured jaw gap clears the object's grasp width.

    Assumes symmetric fingers with joint positions that increase when opening.
    This checks current clearance, not whether a grasp happened earlier. An open
    gripper can pass before grasping; an opening command alone cannot make it pass.

    Args:
        env: Environment supplying measured joint positions through ArenaWorld.
        robot_name: Robot scene entity name.
        gripper_joint_name: Finger joint whose position measures the jaw opening.
        jaw_gap_at_zero_joint_m: Jaw gap when the finger joint position is zero,
            in meters. For fully open gap ``span_m`` and joint position
            ``open_joint_m``, this is ``span_m - 2 * open_joint_m``.
        grasp_width_m: Object width at the grasp, in meters.
        release_clearance_m: Required extra gap beyond the object width, in meters.
            The comparison is strict, so exactly this clearance does not pass.

    Returns:
        Boolean tensor with one result per environment.
    """
    assert math.isfinite(jaw_gap_at_zero_joint_m), "Jaw-gap offset must be finite."
    assert math.isfinite(grasp_width_m) and grasp_width_m > 0.0, "Grasp width must be positive and finite."
    assert (
        math.isfinite(release_clearance_m) and release_clearance_m >= 0.0
    ), "Release clearance must be non-negative and finite."
    measured = env.arena_world.get_joint_position(robot_name, gripper_joint_name)
    # Both fingers move outward by measured, increasing the zero-position gap by twice that amount.
    gap = jaw_gap_at_zero_joint_m + 2.0 * measured
    return gap > grasp_width_m + release_clearance_m
