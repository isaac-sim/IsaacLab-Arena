# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless predicates over the state of articulated objects."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.utils.joint_utils import get_normalized_joint_position


def is_away_from_rest_openness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rest_openness: float,
    min_openness_change: float,
) -> torch.Tensor:
    """Checks if a joint's openness is away from a rest openness.

    Movement in either direction counts: the check is abs(openness - rest_openness) > min_openness_change.
    This reads the current state, so a door that is opened and then closed again returns False.

    Args:
        env: The environment to read the joint state from.
        asset_cfg: The scene entity and joint to read the openness of.
        rest_openness: The openness the movement is measured against, typically the reset openness.
        min_openness_change: How far the openness must change from rest_openness to count as moved.

    Returns:
        One Boolean result per environment.
    """
    openness = get_normalized_joint_position(env, asset_cfg)
    return (openness - rest_openness).abs() > min_openness_change
