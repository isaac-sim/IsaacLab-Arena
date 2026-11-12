# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build actions that keep robots still from each action term's previous action."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.managers import ActionManager, ActionTerm


def get_still_actions(action_manager: ActionManager) -> torch.Tensor:
    """Return actions that keep every robot still, shaped (num_envs, total_action_dim).

    A term that defines get_still_actions(previous_actions) builds its own. Otherwise, a term whose actions
    are changes or velocities holds still at zero, and any other term repeats its previous action, which
    holds still only when that action is an absolute target.

    Args:
        action_manager: Action manager holding the actions applied on the previous step.
    """
    still_actions = action_manager.action.clone()
    first_index = 0
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        indices = slice(first_index, first_index + term.action_dim)
        if hasattr(term, "get_still_actions"):
            still_actions[:, indices] = term.get_still_actions(still_actions[:, indices])
        elif _holds_still_at_zero(term):
            still_actions[:, indices] = 0.0
        first_index += term.action_dim
    return still_actions


def _holds_still_at_zero(term: ActionTerm) -> bool:
    """Check whether a term's actions are changes or velocities, which keep the robot still at zero."""
    from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction, RelativeJointPositionAction
    from isaaclab.envs.mdp.actions.non_holonomic_actions import NonHolonomicAction

    if isinstance(term, (RelativeJointPositionAction, JointVelocityAction, NonHolonomicAction)):
        return True
    # Differential IK sets relative mode on its controller; RMPflow sets it on the term.
    controller_cfg = getattr(term.cfg, "controller", None)
    return bool(getattr(term.cfg, "use_relative_mode", False) or getattr(controller_cfg, "use_relative_mode", False))
