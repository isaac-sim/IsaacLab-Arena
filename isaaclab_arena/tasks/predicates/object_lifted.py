# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Detect lifting relative to the height at first activation."""

from __future__ import annotations

import math
import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg


class ObjectLifted(ManagerTermBase):
    """Capture a reference height on first active evaluation and detect a subsequent rise.

    Configure with TerminationTermCfg using object_name and optional distance (default 0.01 m).
    Use a settling prerequisite to capture the reference at rest. Policy actions continue while
    waiting; earlier motion becomes part of the reference. Each occurrence owns its heights.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        object_name = cfg.params["object_name"]
        distance = cfg.params.get("distance", 1e-2)
        assert isinstance(object_name, str) and object_name, "object_name must be a non-empty scene key."
        assert math.isfinite(distance) and distance > 0, "distance must be finite and positive."
        self._reference_height = torch.full((env.num_envs,), float("nan"), device=env.device)
        self._has_reference_height = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env,
        object_name: str,
        distance: float = 1e-2,
        *,
        active_envs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return where an active object has risen strictly more than distance above its reference.

        Args:
            env: Environment containing the object.
            object_name: Scene key of the object to track.
            distance: Required vertical rise in meters.
            active_envs: Boolean mask of environments to evaluate; defaults to all environments.
        """
        if active_envs is None:
            active_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        current_height = env.arena_world.get_position_w(object_name)[:, 2]
        needs_reference = active_envs & ~self._has_reference_height
        self._reference_height[needs_reference] = current_height[needs_reference]
        self._has_reference_height |= needs_reference
        return active_envs & self._has_reference_height & (current_height > self._reference_height + distance)

    def reset(self, env_ids=None) -> None:
        """Clear reference heights for the selected environments, or all environments when omitted."""
        selected_env_ids = slice(None) if env_ids is None else env_ids
        self._reference_height[selected_env_ids] = float("nan")
        self._has_reference_height[selected_env_ids] = False
