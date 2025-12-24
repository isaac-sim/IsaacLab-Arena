# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import argparse
import gymnasium as gym
import torch
from abc import ABC, abstractmethod
from gymnasium.spaces.dict import Dict as GymSpacesDict

from enum import Enum

from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig
from isaaclab_arena.remote_policy.policy_client import PolicyClient

class PolicyDeployment(Enum):
    LOCAL = "local"
    REMOTE = "remote"

class PolicyBase(ABC):
    def __init__(
        self,
        policy_deployment: PolicyDeployment = PolicyDeployment.LOCAL,
        remote_config: RemotePolicyConfig | None = None
    ) -> None:
        """
        Base class for policies with optional remote deployment.

        Args:
            policy_deployment: "local" (default) or "remote".
            remote_config: Required when policy_deployment == "remote".
        """
        self._policy_deployment = policy_deployment
        self._remote_config = remote_config
        self._policy_client: PolicyClient | None = None

        if self._policy_deployment is PolicyDeployment.REMOTE:
            if self._remote_config is None:
                raise ValueError("Remote deployment requires a RemotePolicyConfig.")

            self._policy_client = PolicyClient(
                config=self._remote_config,
            )

    @property
    def is_remote(self) -> bool:
        return self._policy_deployment is PolicyDeployment.REMOTE

    @property
    def remote_config(self) -> RemotePolicyConfig:
        return self._remote_config

    @property
    def remote_client(self) -> RemotePolicyClient:
        if self._policy_client is None:
            raise RuntimeError("Remote client is not initialized (policy_deployment != 'remote').")
        return self._policy_client

    @abstractmethod
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Compute an action given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        raise NotImplementedError("Function not implemented yet.")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the policy.
        """
        pass

    def set_task_description(self, task_description: str | None) -> str:
        """Set the task description of the task being evaluated."""
        self.task_description = task_description
        return self.task_description

    def has_length(self) -> bool:
        """Check if the policy is based on a recording (i.e. is a dataset-driven policy)."""
        return False

    def length(self) -> int | None:
        """Get the length of the policy (for dataset-driven policies)."""
        pass

    @staticmethod
    @abstractmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add policy-specific arguments to the parser."""
        raise NotImplementedError("Function not implemented yet.")

    @staticmethod
    @abstractmethod
    def from_args(args: argparse.Namespace) -> "PolicyBase":
        """Create a policy from the arguments."""
        raise NotImplementedError("Function not implemented yet.")
    def shutdown_remote(self, kill_server: bool = False) -> None:
        """
        Clean up remote client, and optionally send 'kill' to stop the remote server.

        Args:
            kill_server: If True, send a 'kill' RPC before closing the client.
        """
        if not self.is_remote or self._policy_client is None:
            return
        if kill_server:
            try:
                self._policy_client.call_endpoint("kill", requires_input=False)
            except Exception as exc:
                print(f"[PolicyBase] Failed to send kill to remote server: {exc}")
        self._policy_client.close()
        self._policy_client = None
