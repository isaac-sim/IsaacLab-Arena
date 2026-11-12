# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the actions that keep robots still in environments whose episode finished."""

import torch
from types import SimpleNamespace

import pytest
from isaaclab.envs.mdp.actions.joint_actions import (
    JointPositionAction,
    JointVelocityAction,
    RelativeJointPositionAction,
)
from isaaclab.envs.mdp.actions.non_holonomic_actions import NonHolonomicAction
from isaaclab.managers import ActionTerm

from isaaclab_arena.environments.still_actions import get_still_actions


def _term(term_type: type[ActionTerm], action_dim: int, cfg=None) -> ActionTerm:
    """Return a term of term_type without the simulation assets its constructor needs."""
    stubs = {name: None for name in term_type.__abstractmethods__}
    term = object.__new__(type(term_type.__name__, (term_type,), {**stubs, "action_dim": action_dim}))
    term.cfg = cfg if cfg is not None else SimpleNamespace()
    return term


def _action_manager(previous_actions: torch.Tensor, **terms: ActionTerm):
    return SimpleNamespace(action=previous_actions, active_terms=list(terms), get_term=terms.__getitem__)


PREVIOUS_ACTIONS = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_absolute_targets_repeat_and_changes_or_velocities_are_zero():
    action_manager = _action_manager(
        PREVIOUS_ACTIONS, arm=_term(JointPositionAction, 2), gripper=_term(JointVelocityAction, 1)
    )

    torch.testing.assert_close(get_still_actions(action_manager), torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]))
    # The previous actions are left untouched.
    torch.testing.assert_close(action_manager.action, PREVIOUS_ACTIONS)


@pytest.mark.parametrize(
    "term",
    [
        _term(RelativeJointPositionAction, 3),
        _term(NonHolonomicAction, 3),
        _term(ActionTerm, 3, SimpleNamespace(controller=SimpleNamespace(use_relative_mode=True))),
        _term(ActionTerm, 3, SimpleNamespace(use_relative_mode=True)),
    ],
    ids=["relative_joint_position", "non_holonomic", "relative_differential_ik", "relative_rmpflow"],
)
def test_relative_terms_hold_still_at_zero(term):
    torch.testing.assert_close(get_still_actions(_action_manager(PREVIOUS_ACTIONS, arm=term)), torch.zeros(2, 3))


@pytest.mark.parametrize(
    "cfg",
    [
        SimpleNamespace(controller=SimpleNamespace(use_relative_mode=False)),
        SimpleNamespace(use_relative_mode=False),
        SimpleNamespace(),
    ],
    ids=["absolute_differential_ik", "absolute_rmpflow", "other"],
)
def test_other_terms_repeat_their_previous_action(cfg):
    action_manager = _action_manager(PREVIOUS_ACTIONS, arm=_term(ActionTerm, 3, cfg))

    torch.testing.assert_close(get_still_actions(action_manager), PREVIOUS_ACTIONS)


def test_terms_can_build_their_own_still_actions():
    term = _term(ActionTerm, 2)
    term.get_still_actions = lambda previous_actions: previous_actions * 10
    action_manager = _action_manager(PREVIOUS_ACTIONS, arm=term, gripper=_term(JointVelocityAction, 1))

    torch.testing.assert_close(get_still_actions(action_manager), torch.tensor([[10.0, 20.0, 0.0], [40.0, 50.0, 0.0]]))
