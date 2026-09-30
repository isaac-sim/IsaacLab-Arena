# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Activation and reset contract for predicates with episode state."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.progress_tracking.progress_tracker import _PredicateEvaluation


class StatefulPredicate(ABC):
    """Own one predicate occurrence's state for each environment.

    The runner selects active environments and resets this predicate. Wrappers evaluate
    children through the shared evaluation cache and reset their owned stateful children.
    """

    requires_step_index = False
    """Whether consecutive control-step indices are required by this predicate or any child."""

    @abstractmethod
    def evaluate(self, evaluation: _PredicateEvaluation, active_envs: torch.Tensor) -> torch.Tensor:
        """Return one Boolean per environment, updating state only for active_envs."""

    @abstractmethod
    def reset(self, env_ids: list[int] | torch.Tensor) -> None:
        """Clear episode state for the selected environments, including owned children."""
