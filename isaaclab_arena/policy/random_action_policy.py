# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class RandomActionPolicyCfg(PolicyCfg):
    """Configure reproducible, small random actions."""

    random_seed: int = 42
    """Seed used by the policy's random number generator."""

    action_scale: float = 0.05
    """Absolute random-action scale used when no supported pose observation is available."""

    eef_position_noise: float = 0.01
    """Maximum position offset in metres for each GR1 end effector."""

    hand_joint_noise: float = 0.05
    """Maximum joint-position offset in radians for each GR1 hand joint."""

    def __post_init__(self) -> None:
        assert self.action_scale >= 0.0, "action_scale must be non-negative"
        assert self.eef_position_noise >= 0.0, "eef_position_noise must be non-negative"
        assert self.hand_joint_noise >= 0.0, "hand_joint_noise must be non-negative"


@register_policy
class RandomActionPolicy(PolicyBase[RandomActionPolicyCfg]):
    """Generate bounded random actions, preserving valid GR1 Pink poses when possible."""

    name = "random_action"

    _GR1_PINK_OBSERVATION_KEYS = (
        "left_eef_pos",
        "left_eef_quat",
        "right_eef_pos",
        "right_eef_quat",
        "hand_joint_state",
    )

    def __init__(self, config: RandomActionPolicyCfg):
        """Initialize the policy from its typed configuration.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        """Return the reproducibly seeded generator for a device."""
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.config.random_seed)
            self._generators[key] = generator
        return self._generators[key]

    def _uniform_noise(self, reference: torch.Tensor, scale: float) -> torch.Tensor:
        """Sample uniform noise with the same shape, device, and dtype as a tensor."""
        if scale == 0.0:
            return torch.zeros_like(reference)
        return torch.empty_like(reference).uniform_(-scale, scale, generator=self._generator(reference.device))

    def _gr1_pink_action(self, policy_observation: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Build a safe GR1 Pink action around the currently observed pose."""
        if not all(key in policy_observation for key in self._GR1_PINK_OBSERVATION_KEYS):
            return None

        left_pos = policy_observation["left_eef_pos"]
        left_quat = policy_observation["left_eef_quat"]
        right_pos = policy_observation["right_eef_pos"]
        right_quat = policy_observation["right_eef_quat"]
        hand_joints = policy_observation["hand_joint_state"]

        return torch.cat(
            (
                left_pos + self._uniform_noise(left_pos, self.config.eef_position_noise),
                left_quat,
                right_pos + self._uniform_noise(right_pos, self.config.eef_position_noise),
                right_quat,
                hand_joints + self._uniform_noise(hand_joints, self.config.hand_joint_noise),
            ),
            dim=-1,
        )

    @staticmethod
    def _clamp_to_action_space(actions: torch.Tensor, action_space: gym.Space) -> torch.Tensor:
        """Clamp finite Box bounds while leaving unbounded dimensions unchanged."""
        if not isinstance(action_space, gym.spaces.Box):
            return actions

        low = torch.as_tensor(action_space.low, device=actions.device, dtype=actions.dtype)
        high = torch.as_tensor(action_space.high, device=actions.device, dtype=actions.dtype)
        actions = torch.where(torch.isfinite(low), torch.maximum(actions, low), actions)
        return torch.where(torch.isfinite(high), torch.minimum(actions, high), actions)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Return a small random action compatible with the environment."""
        device = torch.device(env.unwrapped.device)
        policy_observation = observation.get("policy", observation)
        actions = self._gr1_pink_action(policy_observation)

        if actions is None or tuple(actions.shape) != tuple(env.action_space.shape):
            actions = torch.empty(env.action_space.shape, device=device).uniform_(
                -self.config.action_scale,
                self.config.action_scale,
                generator=self._generator(device),
            )

        return self._clamp_to_action_space(actions, env.action_space)
