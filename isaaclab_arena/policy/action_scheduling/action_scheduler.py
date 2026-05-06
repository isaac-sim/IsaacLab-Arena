# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for action scheduling strategies."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Callable


class ActionScheduler(ABC):
    """Translates raw action outputs into per-step actions.

    The policy calls ``get_action(fetch_action_fn)`` at every environment step.
    The scheduler controls when to query the action and how to derive a single
    action from one or more action outputs.
    """

    @abstractmethod
    def get_action(self, fetch_action_tensor_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Return one action per env for the current timestep.

        Args:
            fetch_action_tensor_fn: Callable that queries the action and returns an action tensor.

        Returns:
            Per-step action tensor of shape ``(num_envs, action_dim)``.
        """
        ...
