# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Detect lifting relative to the height at first activation."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.tasks.predicates.stateful_predicate import StatefulPredicate

if TYPE_CHECKING:
    from isaaclab_arena.progress_tracking.progress_tracker import _PredicateEvaluation


class ObjectLifted(ManagerTermBase, StatefulPredicate):
    """Capture a reference height on first active evaluation and detect a subsequent rise.

    Configure with TerminationTermCfg using object_name and optional distance (default 0.01 m).
    Use a settling prerequisite to capture the reference at rest. Policy actions continue while
    waiting; earlier motion becomes part of the reference. Each occurrence owns its heights.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        self._object_name = cfg.params["object_name"]
        self._distance = cfg.params.get("distance", 1e-2)
        assert isinstance(self._object_name, str) and self._object_name, "object_name must be a non-empty scene key."
        assert math.isfinite(self._distance) and self._distance > 0, "distance must be finite and positive."
        self._reference_height = torch.full((env.num_envs,), float("nan"), device=env.device)
        self._has_reference_height = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def evaluate(self, evaluation: _PredicateEvaluation, active_envs: torch.Tensor) -> torch.Tensor:
        current_height = evaluation.env.arena_world.get_position_w(self._object_name)[:, 2]
        needs_reference = active_envs & ~self._has_reference_height
        self._reference_height[needs_reference] = current_height[needs_reference]
        self._has_reference_height |= needs_reference
        return self._has_reference_height & (current_height > self._reference_height + self._distance)

    def reset(self, env_ids: list[int] | torch.Tensor) -> None:
        self._reference_height[env_ids] = float("nan")
        self._has_reference_height[env_ids] = False
