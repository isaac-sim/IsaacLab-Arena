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
    """Translates raw model chunk outputs into per-step actions.

    The policy calls ``get_action(fetch_chunk_fn)`` at every environment step.
    The scheduler controls when to query the model and how to derive a single
    action from one or more model outputs.

    Concrete implementations include:
    - ``ActionChunkScheduler``: buffer one chunk, step through it sequentially,
      refetch when exhausted.
    - ``TemporalEnsemblingScheduler``: always query the model, blend overlapping
      chunks with exponential decay weights (ACT-style).
    - ``PassThroughScheduler``: always query the model, return the first action
      in the chunk.
    """

    @abstractmethod
    def get_action(self, fetch_chunk_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Return one action per env for the current timestep.

        Args:
            fetch_chunk_fn: Callable that queries the model and returns a chunk
                tensor of shape ``(num_envs, horizon, action_dim)``.

        Returns:
            Action tensor of shape ``(num_envs, action_dim)``.
        """
        ...

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset scheduler state for the given envs (all envs if None)."""
        ...
