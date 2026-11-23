# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation and reward terms over one robot's action terms.

Isaac Lab's ``last_action`` and ``action_rate_l2`` read the whole action tensor. A robot whose
actions are several named terms in a shared environment selects its own terms with these.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def robot_last_action(env: ManagerBasedEnv, action_names: Sequence[str]) -> torch.Tensor:
    """Return the raw actions of the named action terms, concatenated in the given order.

    Args:
        env: The environment.
        action_names: Names of the robot's action terms.
    """
    return torch.cat([env.action_manager.get_term(name).raw_actions for name in action_names], dim=-1)


def robot_action_rate_l2(env: ManagerBasedRLEnv, action_names: Sequence[str]) -> torch.Tensor:
    """Penalize the squared change of the named action terms' actions since the previous step.

    Args:
        env: The environment.
        action_names: Names of the robot's action terms.
    """
    manager = env.action_manager
    changes = []
    column = 0
    for name, width in zip(manager.active_terms, manager.action_term_dim):
        if name in action_names:
            changes.append(manager.action[:, column : column + width] - manager.prev_action[:, column : column + width])
        column += width
    assert len(changes) == len(action_names), f"Action terms {action_names} must all be active"
    return torch.sum(torch.square(torch.cat(changes, dim=-1)), dim=1)
