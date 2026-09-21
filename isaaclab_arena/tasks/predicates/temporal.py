# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Temporal predicate requirements with runtime state owned by ProgressObjectiveRunner."""

import torch
from collections.abc import Callable
from dataclasses import dataclass

from isaaclab.managers import TerminationTermCfg


# Isaac Lab prepares nested configuration fields in place, so this dataclass must remain mutable.
@dataclass
class TrueForConsecutiveStepsCfg:
    """Declare how long an instantaneous predicate must remain true.

    ProgressObjectiveRunner creates and owns a _TrueForConsecutiveSteps runtime instance
    for each configured occurrence.
    The predicate returns one Boolean per environment without maintaining a streak itself.
    ProgressObjectiveRunner resets the consecutive-step counter, not the wrapped predicate.
    The wrapped predicate must not depend on this reset path to clear its own episode state.

    Require the cube to remain below the velocity thresholds for ten consecutive control steps::

        from functools import partial
        from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds

        resting = partial(objects_below_velocity_thresholds, object_names=["cube"])
        TrueForConsecutiveStepsCfg(
            predicate=resting,
            required_steps=10,
        )
    """

    predicate: Callable | TerminationTermCfg
    """Instantaneous check; a callable class in TerminationTermCfg can be initialized with the environment."""

    required_steps: int
    """Positive number of consecutive qualifying control steps."""

    def __post_init__(self):
        assert isinstance(self.predicate, TerminationTermCfg) or (
            callable(self.predicate) and not isinstance(self.predicate, type)
        ), "predicate must be a configured instantaneous callable or TerminationTermCfg."
        assert (
            isinstance(self.required_steps, int)
            and not isinstance(self.required_steps, bool)
            and self.required_steps > 0
        ), "required_steps must be a positive integer."


class _TrueForConsecutiveSteps:
    """Runtime state for TrueForConsecutiveStepsCfg, owned by ProgressObjectiveRunner.

    ProgressObjectiveRunner supplies predicate results, selects active environments,
    and resets this requirement. This class does not evaluate predicates or track step indices.
    """

    def __init__(self, *, predicate: Callable, required_steps: int, num_envs: int, device):
        self.predicate = predicate
        """Instantaneous callable prepared by ProgressObjectiveRunner."""
        self.required_steps = required_steps
        self._consecutive_true_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, predicate_results: torch.Tensor, active_envs: torch.Tensor) -> torch.Tensor:
        """Update active environments' streaks and return which have reached required_steps."""
        next_counts = torch.where(
            predicate_results,
            (self._consecutive_true_steps + 1).clamp(max=self.required_steps),
            0,
        )
        self._consecutive_true_steps[active_envs] = next_counts[active_envs]
        return self._consecutive_true_steps >= self.required_steps

    def reset(self, env_ids: list[int] | torch.Tensor) -> None:
        """Clear the streaks for the selected environments."""
        self._consecutive_true_steps[env_ids] = 0
