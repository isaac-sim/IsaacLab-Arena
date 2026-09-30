# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Consecutive-step predicate requirements and their per-occurrence state."""

from __future__ import annotations

import functools
import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted

if TYPE_CHECKING:
    from isaaclab_arena.progress_tracking.progress_tracker import _PredicateEvaluation


@dataclass
class TrueForConsecutiveStepsCfg:
    """Require a predicate to remain true for consecutive active control steps.

    Each occurrence owns its counter and its configured child. Reusing a configuration
    creates separate instances; reusing an ordinary callable shares its per-step evaluation.
    """

    predicate: Callable | TerminationTermCfg | TrueForConsecutiveStepsCfg
    """Configured callable or nested consecutive-step requirement."""

    required_steps: int
    """Positive number of consecutive qualifying control steps."""

    def __post_init__(self):
        predicate_function = self.predicate.func if isinstance(self.predicate, functools.partial) else self.predicate
        assert not isinstance(predicate_function, ObjectLifted) and (
            isinstance(self.predicate, (TerminationTermCfg, TrueForConsecutiveStepsCfg))
            or (callable(self.predicate) and not isinstance(self.predicate, type))
        ), "predicate must be a callable, TerminationTermCfg, or TrueForConsecutiveStepsCfg."
        assert (
            isinstance(self.required_steps, int)
            and not isinstance(self.required_steps, bool)
            and self.required_steps > 0
        ), "required_steps must be a positive integer."


class _TrueForConsecutiveSteps:
    """Count consecutive true steps and reset owned lift or consecutive-step children."""

    def __init__(self, *, predicate: Callable | _TrueForConsecutiveSteps, required_steps: int, num_envs: int, device):
        self.predicate = predicate
        self.required_steps = required_steps
        self._consecutive_true_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def evaluate(self, evaluation: _PredicateEvaluation, active_envs: torch.Tensor) -> torch.Tensor:
        predicate_results = evaluation.evaluate(self.predicate, active_envs)
        next_counts = torch.where(
            predicate_results,
            (self._consecutive_true_steps + 1).clamp(max=self.required_steps),
            0,
        )
        self._consecutive_true_steps[active_envs] = next_counts[active_envs]
        return self._consecutive_true_steps >= self.required_steps

    def reset(self, env_ids: list[int] | torch.Tensor) -> None:
        self._consecutive_true_steps[env_ids] = 0
        predicate = self.predicate.func if isinstance(self.predicate, functools.partial) else self.predicate
        if isinstance(predicate, (ObjectLifted, _TrueForConsecutiveSteps)):
            predicate.reset(env_ids)
