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

    Without an episode limit every finished episode is replaced at once, so the list is empty.

    Args:
        env: Arena environment, optionally gym-wrapped.
    """
    return (~env.unwrapped.active_episode_mask).nonzero(as_tuple=True)[0].tolist()


_UNSUPPORTED_HOLD_TERMS_WARNED: set[str] = set()


def hold_action_from_action_manager(env: gym.Env) -> torch.Tensor | None:
    """Return the action that keeps every env's robot where it is for one control step, or None.

    Each action term contributes its slice of the ``(num_envs, total_action_dim)`` result:

    - absolute joint position terms get the current joint positions, mapped back through the
      term's scale and offset, so the drive target is the current pose;
    - absolute inverse-kinematics terms get the end effector's current pose in the base frame, mapped
      back the same way, so an env that never acted holds where it stands instead of being sent to the
      base origin;
    - relative joint position, joint velocity, non-holonomic and relative inverse-kinematics
      terms get zero, their "stay in place" command;
    - binary joint terms repeat the raw action they applied last, which is zero right after a reset.

    For any other term type there is no hold action, since repeating its last command would keep the
    robot moving: the function prints a warning once per term type and returns None, and callers keep
    the behaviour they had without holding.

    Args:
        env: Environment whose action manager defines the action layout.
    """
    from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction
    from isaaclab.envs.mdp.actions.joint_actions import (
        JointPositionAction,
        JointVelocityAction,
        RelativeJointPositionAction,
    )
    from isaaclab.envs.mdp.actions.non_holonomic_actions import NonHolonomicAction
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    action_manager = env.unwrapped.action_manager
    term_hold_actions = []
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        if isinstance(term, (RelativeJointPositionAction, JointVelocityAction, NonHolonomicAction)):
            term_hold_actions.append(torch.zeros_like(term.raw_actions))
        elif isinstance(term, JointPositionAction):
            joint_pos = term._asset.data.joint_pos.torch[:, term._joint_ids]
            term_hold_actions.append((joint_pos - term._offset) / term._scale)
        elif isinstance(term, DifferentialInverseKinematicsAction) and term.cfg.controller.use_relative_mode:
            term_hold_actions.append(torch.zeros_like(term.raw_actions))
        elif isinstance(term, DifferentialInverseKinematicsAction):
            ee_pos_b, ee_quat_b = term._compute_frame_pose()
            if term.cfg.controller.command_type == "position":
                ee_command = ee_pos_b
            else:
                ee_command = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
            term_hold_actions.append((ee_command - term._offset) / term._scale)
        elif isinstance(term, BinaryJointAction):
            term_hold_actions.append(term.raw_actions.clone())
        else:
            term_type = type(term).__name__
            if term_type not in _UNSUPPORTED_HOLD_TERMS_WARNED:
                _UNSUPPORTED_HOLD_TERMS_WARNED.add(term_type)
                print(
                    f"[idle_envs] No hold action is defined for action term '{term_name}' of type {term_type};"
                    " idle environments keep acting as if they were not idle."
                )
            return None
    return torch.cat(term_hold_actions, dim=-1)
