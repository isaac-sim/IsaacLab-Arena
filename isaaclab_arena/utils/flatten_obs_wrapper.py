# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrapper to flatten dict observations for RSL-RL compatibility."""

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class FlattenDictObsWrapper(gym.Wrapper):
    """Wraps an environment to flatten dict observations into single tensors.

    This wrapper is useful when the environment has `concatenate_terms=False` 
    (returning dict observations) but you want to use RSL-RL which requires 
    flattened 1D observations.

    The wrapper:
    1. Intercepts observations from reset() and step()
    2. For each observation group that is a dict, concatenates all values into a single tensor
    3. Returns flattened observations compatible with RSL-RL

    Example:
        >>> env = gym.make(env_name, cfg=env_cfg)
        >>> env = FlattenDictObsWrapper(env)  # Add this before RslRlVecEnvWrapper
        >>> env = RslRlVecEnvWrapper(env)
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self._obs_keys_order: dict[str, list[str]] = {}
        self._obs_shapes: dict[str, dict[str, tuple]] = {}

    def _flatten_obs_group(self, obs_group_name: str, obs_group: dict | torch.Tensor) -> torch.Tensor:
        """Flatten a single observation group.

        Args:
            obs_group_name: Name of the observation group (e.g., "policy", "task_obs").
            obs_group: Either a dict of tensors or already a tensor.

        Returns:
            Flattened tensor of shape (num_envs, total_dim).
        """
        # If already a tensor, return as-is
        if isinstance(obs_group, torch.Tensor):
            return obs_group

        # If dict, flatten it
        if isinstance(obs_group, dict):
            # Store key order for this group (for potential unflattening later)
            if obs_group_name not in self._obs_keys_order:
                self._obs_keys_order[obs_group_name] = list(obs_group.keys())
                self._obs_shapes[obs_group_name] = {
                    k: v.shape[1:] for k, v in obs_group.items()
                }

            # Flatten each tensor and concatenate
            flattened_parts = []
            for key in self._obs_keys_order[obs_group_name]:
                tensor = obs_group[key]
                # Flatten all dimensions except batch dimension
                flattened = tensor.reshape(tensor.shape[0], -1)
                flattened_parts.append(flattened)

            return torch.cat(flattened_parts, dim=-1)

        raise TypeError(f"Unexpected observation type: {type(obs_group)}")

    def _flatten_obs_dict(self, obs_dict: dict) -> dict:
        """Flatten all observation groups in the observation dict.

        Args:
            obs_dict: Dictionary of observation groups.

        Returns:
            Dictionary with all dict groups flattened to tensors.
        """
        flattened = {}
        for group_name, group_obs in obs_dict.items():
            flattened[group_name] = self._flatten_obs_group(group_name, group_obs)
        return flattened

    def reset(self, **kwargs):
        """Reset the environment and flatten observations.

        Returns:
            Tuple of (flattened_obs_dict, info).
        """
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs_dict(obs_dict), info

    def step(self, action):
        """Step the environment and flatten observations.

        Args:
            action: Action to take.

        Returns:
            Tuple of (flattened_obs_dict, reward, terminated, truncated, info).
        """
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs_dict(obs_dict), reward, terminated, truncated, info

    @property
    def obs_keys_order(self) -> dict[str, list[str]]:
        """Get the order of keys for each observation group.

        Useful for understanding which features correspond to which observation terms.
        """
        return self._obs_keys_order

    @property
    def obs_shapes(self) -> dict[str, dict[str, tuple]]:
        """Get the original shapes of each observation term.

        Useful for unflattening observations if needed.
        """
        return self._obs_shapes


class FlattenObsRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL VecEnv wrapper that flattens dict observations.

    This wrapper extends RslRlVecEnvWrapper to handle environments with
    `concatenate_terms=False` by flattening dict observations into tensors.

    Use this instead of RslRlVecEnvWrapper when your environment returns
    dict observations.
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            clip_actions: Clip actions to this range. If None, no clipping is done.
        """
        super().__init__(env, clip_actions)
        self._obs_keys_order: dict[str, list[str]] = {}
        self._obs_shapes: dict[str, dict[str, tuple]] = {}
        # Cache the task description
        self._task_description: str | None = self._get_task_description()

    def _get_task_description(self) -> str | None:
        """Get the task description from the task object."""
        try:
            return self.unwrapped.cfg.isaaclab_arena_env.task.task_description
        except AttributeError:
            return None

    @property
    def task_description(self) -> str | None:
        """Get the task description string."""
        return self._task_description

    def _flatten_obs_group(self, obs_group_name: str, obs_group: dict | torch.Tensor) -> torch.Tensor:
        """Flatten a single observation group."""
        # If already a tensor, return as-is
        if isinstance(obs_group, torch.Tensor):
            return obs_group

        # If dict, flatten it
        if isinstance(obs_group, dict):
            # Store key order for this group (for potential unflattening later)
            if obs_group_name not in self._obs_keys_order:
                self._obs_keys_order[obs_group_name] = list(obs_group.keys())
                self._obs_shapes[obs_group_name] = {
                    k: v.shape[1:] for k, v in obs_group.items()
                }

            # Flatten each tensor and concatenate
            flattened_parts = []
            for key in self._obs_keys_order[obs_group_name]:
                tensor = obs_group[key]
                # Flatten all dimensions except batch dimension
                flattened = tensor.reshape(tensor.shape[0], -1)
                flattened_parts.append(flattened)

            return torch.cat(flattened_parts, dim=-1)

        raise TypeError(f"Unexpected observation type: {type(obs_group)}")

    def _flatten_obs_dict(self, obs_dict: dict) -> dict:
        """Flatten all observation groups in the observation dict."""
        flattened = {}
        for group_name, group_obs in obs_dict.items():
            flattened[group_name] = self._flatten_obs_group(group_name, group_obs)
        return flattened

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset the environment and flatten observations."""
        obs_dict, extras = self.env.reset()
        flattened = self._flatten_obs_dict(obs_dict)
        # Add task description to extras
        if self._task_description is not None:
            extras["task_description"] = self._task_description
        return TensorDict(flattened, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment (flattened)."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        flattened = self._flatten_obs_dict(obs_dict)
        return TensorDict(flattened, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and flatten observations."""
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # add task description to extras
        if self._task_description is not None:
            extras["task_description"] = self._task_description
        # flatten observations
        flattened = self._flatten_obs_dict(obs_dict)
        return TensorDict(flattened, batch_size=[self.num_envs]), rew, dones, extras
