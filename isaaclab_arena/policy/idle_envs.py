# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Skip inference for environments that finished their last episode.

With an episode limit, the Arena env stops replacing finished episodes once the limit is reached but keeps
stepping those envs until the remaining episodes end. Their actions are discarded, so a policy can skip
inference for them and hold their robot in place instead.
"""

from __future__ import annotations

import gymnasium as gym
import torch


def idle_env_ids(env: gym.Env) -> list[int]:
    """Return the envs that finished their last episode and were not assigned a new one.

    Without an episode limit every finished episode is replaced at once, so the list is empty, as it is for
    an env that does not track episode assignment.

    Args:
        env: Environment, optionally gym-wrapped.
    """
    active_episode_mask = getattr(env.unwrapped, "active_episode_mask", None)
    if active_episode_mask is None:
        return []
    return (~active_episode_mask).nonzero(as_tuple=True)[0].tolist()


def hold_action_from_action_manager(env: gym.Env) -> torch.Tensor:
    """Return the action that keeps every env's robot where it is for one control step.

    Each action term contributes its slice of the ``(num_envs, total_action_dim)`` result:

    - absolute joint position terms get the current joint positions, mapped back through the
      term's scale and offset, so the drive target is the current pose;
    - relative joint position terms and relative inverse-kinematics terms get zero, their
      "stay in place" command;
    - every other term, such as a binary gripper or an absolute pose target, repeats the raw
      action it applied last, which is zero right after a reset.

    Args:
        env: Environment whose action manager defines the action layout.
    """
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, RelativeJointPositionAction
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    action_manager = env.unwrapped.action_manager
    term_hold_actions = []
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        if isinstance(term, RelativeJointPositionAction):
            term_hold_actions.append(torch.zeros_like(term.raw_actions))
        elif isinstance(term, JointPositionAction):
            joint_pos = term._asset.data.joint_pos.torch[:, term._joint_ids]
            term_hold_actions.append((joint_pos - term._offset) / term._scale)
        elif isinstance(term, DifferentialInverseKinematicsAction) and term.cfg.controller.use_relative_mode:
            term_hold_actions.append(torch.zeros_like(term.raw_actions))
        else:
            term_hold_actions.append(term.raw_actions.clone())
    return torch.cat(term_hold_actions, dim=-1)
