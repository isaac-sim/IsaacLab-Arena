# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment adapters between Arena observations/actions and GR00T state/action dicts."""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any

from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode


class Gr00tEmbodimentAdapter(ABC):
    """Translates one embodiment's Arena observations and actions to and from GR00T's state and action dicts.

    The adapter owns the embodiment's state units and action layout. Cameras, language, and the request
    format stay in the shared GR00T pipeline.
    """

    action_dim: int
    """Width of one env action produced by build_action_np."""

    @abstractmethod
    def extract_state_np(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return GR00T state arrays keyed by state modality key, each with shape (num_envs, dim)."""

    @abstractmethod
    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        """Convert GR00T's action dict to env actions with shape (num_envs, horizon, action_dim)."""

    @abstractmethod
    def build_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Return the (num_envs, action_dim) action that holds each robot at its current state."""


def create_gr00t_embodiment_adapter(policy_config: Gr00tClosedloopPolicyCfg) -> Gr00tEmbodimentAdapter:
    """Create the embodiment adapter selected by the policy config's task mode."""
    # Imports are deferred so adapter modules can import Gr00tEmbodimentAdapter without a cycle.
    task_mode = TaskMode(policy_config.task_mode_name)
    if task_mode == TaskMode.DROID_MANIPULATION:
        from isaaclab_arena_gr00t.policy.embodiment_adapters.droid_adapter import Gr00tDroidAdapter

        return Gr00tDroidAdapter()
    if task_mode == TaskMode.G1_LOCOMANIPULATION:
        from isaaclab_arena_gr00t.policy.embodiment_adapters.g1_wbc_adapter import Gr00tG1WbcAdapter

        return Gr00tG1WbcAdapter(policy_config)
    if task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
        from isaaclab_arena_gr00t.policy.embodiment_adapters.gr1_arms_adapter import Gr00tGr1ArmsAdapter

        return Gr00tGr1ArmsAdapter(policy_config)
    raise ValueError(f"Unsupported task mode: {task_mode}")
