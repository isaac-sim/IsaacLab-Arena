# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_arena.affordances.openable import Openable

def openable_distance_to_target_openness(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
) -> torch.Tensor:
    diff = openable_object.openable_threshold-openable_object.get_openness(env)
    return -torch.clamp(diff, min=0.0)
