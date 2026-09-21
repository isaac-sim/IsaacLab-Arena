# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture a settled reference height and check lifting against it."""

from __future__ import annotations

import math
import torch

from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.tasks.predicates.consecutive import ConsecutivePredicate
from isaaclab_arena.tasks.predicates.object_settling import (
    DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
    DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    compute_objects_settled_mask,
)

DEFAULT_INITIAL_SETTLING_STEPS = 5


class ObjectSettledWithReference(ConsecutivePredicate):
    """Capture an object's reference height after sustained settling, once per episode.

    ProgressTracker updates active environments and resets the reference on episode reset.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        object_name = cfg.params["object_name"]
        self._object_name = object_name
        assert isinstance(object_name, str) and object_name, "object_name must be a non-empty scene key."
        for parameter_name, default in (
            ("lin_vel_threshold", DEFAULT_LINEAR_VELOCITY_THRESHOLD),
            ("ang_vel_threshold", DEFAULT_ANGULAR_VELOCITY_THRESHOLD),
        ):
            value = cfg.params.get(parameter_name, default)
            assert math.isfinite(value) and value > 0, f"{parameter_name} must be finite and positive."
        self._reference_height = torch.full((env.num_envs,), float("nan"), device=env.device)
        self._has_reference_height = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_settling_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env,
        object_name: str,
        consecutive_steps: int,
        lin_vel_threshold: float = DEFAULT_LINEAR_VELOCITY_THRESHOLD,
        ang_vel_threshold: float = DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return where a settled reference has been captured."""
        # Construction consumes this parameter; managed terms retain it in their calling signature.
        del consecutive_steps
        if active_mask is None:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        current_height = env.arena_world.get_position_w(object_name)[:, 2]
        needs_reference = active_mask & ~self._has_reference_height
        update_settling = needs_reference & (self._last_settling_step != env.episode_length_buf)
        if bool(update_settling.any()):
            below_thresholds = compute_objects_settled_mask(
                env.arena_world, env.scene, [object_name], lin_vel_threshold, ang_vel_threshold
            )
            newly_settled = self._update_consecutive_and_get_completion_mask(below_thresholds, update_settling)
            self._reference_height[newly_settled] = current_height[newly_settled]
            self._has_reference_height |= newly_settled
            self._last_settling_step[update_settling] = env.episode_length_buf[update_settling]
        return active_mask & self._has_reference_height

    def is_above_reference(self, env, distance: float) -> torch.Tensor:
        """Read whether the object is above its captured height without updating settling."""
        assert math.isfinite(distance) and distance > 0, "distance must be finite and positive."
        current_height = env.arena_world.get_position_w(self._object_name)[:, 2]
        return self._has_reference_height & (current_height > self._reference_height + distance)

    def reset(self, env_ids=None) -> None:
        """Clear the settling streak and lift reference for restarted environments."""
        super().reset(env_ids)
        selected_envs = slice(None) if env_ids is None else env_ids
        self._reference_height[selected_envs] = float("nan")
        self._has_reference_height[selected_envs] = False
        self._last_settling_step[selected_envs] = -1


def object_lifted(env, settled_reference: TerminationTermCfg, distance: float = 1e-2) -> torch.Tensor:
    """Check height against the reference captured by the settling prerequisite."""
    return settled_reference.func.is_above_reference(env, distance)
