# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_arena.affordances.openable import Openable

def openable_openness_increment(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
) -> torch.Tensor:
    """
    Reward based on the increment (change) in openness from the last timestep.
    Positive reward when openness increases (door opens more).
    """
    current_openness = openable_object.get_openness(env)
    
    # Initialize previous openness storage if not exists
    storage_key = f"_prev_openness_{id(openable_object)}"
    if not hasattr(env, storage_key):
        setattr(env, storage_key, current_openness.clone())
    
    prev_openness = getattr(env, storage_key)
    
    # Compute increment (positive when opening)
    increment = current_openness - prev_openness

    reward = torch.minimum(increment, openable_object.openable_threshold - prev_openness)

    # Update stored value for next timestep
    setattr(env, storage_key, current_openness.clone())
    
    # Reset previous value on episode reset
    reset_mask = env.reset_buf.bool() if hasattr(env, 'reset_buf') else None
    if reset_mask is not None and reset_mask.any():
        stored = getattr(env, storage_key)
        stored[reset_mask] = current_openness[reset_mask]

    return reward
