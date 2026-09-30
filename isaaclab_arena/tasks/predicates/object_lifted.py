# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Detect lifting relative to the height at first activation."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.tasks.predicates.stateful_predicate import StatefulPredicate, StatefulPredicateCfg

if TYPE_CHECKING:
    from isaaclab_arena.progress_tracking.predicate_runtime import PredicateEvaluation, PredicateFactory


@dataclass
class ObjectLiftedCfg(StatefulPredicateCfg):
    """Detect a rise above the height captured on this occurrence's first active evaluation.

    Use a settling prerequisite when the reference should be captured at rest. Policy actions
    continue while waiting; any earlier motion becomes part of the captured reference height.
    """

    object_name: str
    """Scene key of the object to track."""

    distance: float = 1e-2
    """Required vertical rise in meters; success requires strictly more than this distance."""

    def __post_init__(self):
        assert isinstance(self.object_name, str) and self.object_name, "object_name must be a non-empty scene key."
        assert math.isfinite(self.distance) and self.distance > 0, "distance must be finite and positive."

    def create_runtime(self, factory: PredicateFactory) -> StatefulPredicate:
        return _ObjectLifted(self, factory.num_envs, factory.device)


class _ObjectLifted(StatefulPredicate):
    """Own one lift occurrence's reference height for each environment."""

    def __init__(self, cfg: ObjectLiftedCfg, num_envs: int, device):
        super().__init__(cfg.describe())
        self._object_name = cfg.object_name
        self._distance = cfg.distance
        self._reference_height = torch.full((num_envs,), float("nan"), device=device)
        self._has_reference_height = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def evaluate(self, evaluation: PredicateEvaluation, active_envs: torch.Tensor) -> torch.Tensor:
        current_height = evaluation.env.arena_world.get_position_w(self._object_name)[:, 2]
        needs_reference = active_envs & ~self._has_reference_height
        self._reference_height[needs_reference] = current_height[needs_reference]
        self._has_reference_height |= needs_reference
        return self._has_reference_height & (current_height > self._reference_height + self._distance)

    def reset(self, env_ids: list[int] | torch.Tensor) -> None:
        self._reference_height[env_ids] = float("nan")
        self._has_reference_height[env_ids] = False
