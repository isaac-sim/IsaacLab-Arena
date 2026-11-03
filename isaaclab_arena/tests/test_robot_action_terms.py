# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test that per-robot action terms read only the named robot's actions."""

import torch
from types import SimpleNamespace

from isaaclab_arena.terms.actions import robot_action_rate_l2, robot_last_action


def test_robot_last_action_concatenates_named_terms_in_given_order():
    terms = {
        "left_arm_action": SimpleNamespace(raw_actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        "left_gripper_action": SimpleNamespace(raw_actions=torch.tensor([[5.0], [6.0]])),
        "right_arm_action": SimpleNamespace(raw_actions=torch.tensor([[90.0], [91.0]])),
    }
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=terms.__getitem__))

    actions = robot_last_action(env, ("left_gripper_action", "left_arm_action"))

    assert torch.equal(actions, torch.tensor([[5.0, 1.0, 2.0], [6.0, 3.0, 4.0]]))


def test_robot_action_rate_ignores_other_robots_columns():
    # Columns: left arm (2), right arm (1), left gripper (1). Only the right arm moves far.
    manager = SimpleNamespace(
        active_terms=["left_arm_action", "right_arm_action", "left_gripper_action"],
        action_term_dim=[2, 1, 1],
        action=torch.tensor([[1.0, 2.0, 50.0, 3.0]]),
        prev_action=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    env = SimpleNamespace(action_manager=manager)

    penalty = robot_action_rate_l2(env, ("left_arm_action", "left_gripper_action"))

    assert torch.equal(penalty, torch.tensor([1.0 + 4.0 + 4.0]))
