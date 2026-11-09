# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model inference latency in simulation time.

A policy's ``inference_latency_ms`` is the time its deployment needs between capturing an
observation and applying the first action computed from it. Arena converts it to a whole number of
control steps and delays actions by that many steps, so a rollout's result depends on the configured
latency and the seeds, never on how fast the simulation or the policy server happens to run.
"""

from __future__ import annotations

import gymnasium as gym
import math
import torch
from collections.abc import Sequence


def inference_latency_steps(inference_latency_ms: float, control_step_dt_s: float) -> int:
    """Return the number of control steps that cover ``inference_latency_ms``.

    Rounds up so the modeled latency is never shorter than the configured one.

    Args:
        inference_latency_ms: Observation-to-first-action latency in milliseconds.
        control_step_dt_s: Duration of one control step in seconds.
    """
    assert inference_latency_ms >= 0.0, f"inference_latency_ms must be non-negative, got {inference_latency_ms}"
    assert control_step_dt_s > 0.0, f"control step must be positive, got {control_step_dt_s}"
    return math.ceil(inference_latency_ms / (1000.0 * control_step_dt_s))


def idle_env_ids(env: gym.Env) -> list[int]:
    """Return the envs that finished their last episode and were not assigned a new one.

    With an episode limit, the Arena env stops replacing finished episodes once the limit is reached,
    but keeps stepping those envs until the remaining episodes complete. Their actions are discarded,
    so a policy can skip inference for them and hold their robot in place. Without an episode limit,
    or for an env that does not track episode assignment, every env is active and the list is empty.

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


class PerStepInferenceLatency:
    """Delay a per-step action stream by a fixed number of control steps.

    The action computed from the observation of step ``t`` is applied at step ``t + latency_steps``.
    Until an env has produced that many actions since its last reset, it applies the hold action.
    This gives policies that infer one action per step the same timing as chunked policies under
    the same latency.
    """

    def __init__(self, num_envs: int, latency_steps: int, device: str | torch.device) -> None:
        """Create the delay line.

        Args:
            num_envs: Number of environments in the batch.
            latency_steps: Control steps between an observation and the action computed from it.
            device: Device the action tensors live on.
        """
        from isaaclab.utils.buffers import DelayBuffer

        assert latency_steps >= 1, f"latency_steps must be at least 1, got {latency_steps}"
        self.latency_steps = latency_steps
        self._delay_buffer = DelayBuffer(latency_steps, num_envs, device=str(device))
        self._delay_buffer.set_time_lag(latency_steps)
        self.hold_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        """Number of steps each env applied the hold action, summed over the run."""

    def delay(self, action: torch.Tensor, hold_action: torch.Tensor) -> torch.Tensor:
        """Record this step's action and return the one due at this step.

        Args:
            action: ``(num_envs, action_dim)`` actions computed from this step's observation.
            hold_action: ``(num_envs, action_dim)`` actions for envs whose delayed action is not available yet.
        """
        is_ready = self._delay_buffer.num_pushes >= self.latency_steps
        delayed_action = self._delay_buffer.compute(action)
        self.hold_steps += (~is_ready).long()
        return torch.where(is_ready.unsqueeze(-1), delayed_action, hold_action.to(delayed_action))

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Discard the recorded actions of the given envs (all when None) so they hold again."""
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.reshape(-1).tolist()
        self._delay_buffer.reset(env_ids)
