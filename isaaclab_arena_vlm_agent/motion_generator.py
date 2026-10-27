# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Motion generation contracts for goal-driven VLM policies."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class MotionGenerator(ABC):
    """Generate commands toward goals using live Arena scene state."""

    @abstractmethod
    def goal_reached(self, goal: np.ndarray, env, env_id: int) -> bool:
        """Check goal completion against the current measured state.

        Args:
            goal: Validated goal for one environment.
            env: Arena environment, optionally gym-wrapped.
            env_id: Environment index within the batched world state.

        Returns:
            Whether the goal has been reached.
        """

    @abstractmethod
    def compute_command(self, goal: np.ndarray, env, env_id: int) -> np.ndarray:
        """Compute one command toward the goal using current measured state.

        Args:
            goal: Validated goal for one environment.
            env: Arena environment, optionally gym-wrapped.
            env_id: Environment index within the batched world state.

        Returns:
            A finite one-dimensional command vector for the action adapter.
        """
