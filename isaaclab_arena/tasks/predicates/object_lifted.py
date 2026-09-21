# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Detect lifting relative to the height at first activation."""

from __future__ import annotations

import math
import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

DEFAULT_INITIAL_SETTLING_STEPS = 5


class ObjectLifted(ManagerTermBase):
    """Capture height on first active evaluation and detect a subsequent rise.

    Place a settling prerequisite before this predicate when the reference must be at rest.
    ProgressTracker forwards active masks and episode resets.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        object_name = cfg.params["object_name"]
        assert isinstance(object_name, str) and object_name, "object_name must be a non-empty scene key."
        distance = cfg.params.get("distance", 1e-2)
        assert math.isfinite(distance) and distance > 0, "distance must be finite and positive."
        self._reference_height = torch.full((env.num_envs,), float("nan"), device=env.device)
        self._has_reference_height = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env, object_name: str, distance: float = 1e-2, active_mask=None) -> torch.Tensor:
        """Capture newly active heights and return where the object has risen by more than distance."""
        if active_mask is None:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        current_height = env.arena_world.get_position_w(object_name)[:, 2]
        needs_reference = active_mask & ~self._has_reference_height
        self._reference_height[needs_reference] = current_height[needs_reference]
        self._has_reference_height |= needs_reference
        return active_mask & self._has_reference_height & (current_height > self._reference_height + distance)

    def reset(self, env_ids=None) -> None:
        """Clear the lift reference for restarted environments."""
        selected_envs = slice(None) if env_ids is None else env_ids
        self._reference_height[selected_envs] = float("nan")
        self._has_reference_height[selected_envs] = False
