# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode recorder terms that record how much an object or a joint moved during the episode."""

from __future__ import annotations

import math
import torch
from typing import Any

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg


def _finite_or_none(value: torch.Tensor) -> float | None:
    """Return value as a float, or None if it is NaN or infinite (a JSON record cannot hold either)."""
    number = float(value)
    return number if math.isfinite(number) else None


class ObjectPeakSpeed(EpisodeRecorderTerm):
    """Records the highest linear speed [m/s] an object reached during the episode.

    The speed is None if the simulation produced a non-finite velocity during the episode.
    """

    def __init__(self, cfg: EpisodeRecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._object_name = cfg.params["object_name"]
        self._peak_linear_speed = torch.zeros(env.num_envs, device=env.device)

    def post_step(self, env) -> None:
        linear_velocity_w = env.arena_world.get_mean_linear_velocity_w(self._object_name)
        linear_speed = torch.linalg.norm(linear_velocity_w, dim=-1)
        torch.maximum(self._peak_linear_speed, linear_speed, out=self._peak_linear_speed)

    def reset(self, env_ids=None) -> None:
        self._peak_linear_speed[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, env_id: int, object_name: str) -> dict[str, Any]:
        return {"object": object_name, "peak_linear_speed": _finite_or_none(self._peak_linear_speed[env_id])}


class JointOpennessRange(EpisodeRecorderTerm):
    """Records the lowest and highest openness of an openable object's joint during the episode.

    An openness is None if the simulation produced a non-finite value during the episode.
    """

    def __init__(self, cfg: EpisodeRecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._openable_object = cfg.params["openable_object"]
        self._min_openness = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._max_openness = torch.full((env.num_envs,), float("-inf"), device=env.device)

    def post_step(self, env) -> None:
        openness = self._openable_object.get_openness(env).reshape(env.num_envs)
        torch.minimum(self._min_openness, openness, out=self._min_openness)
        torch.maximum(self._max_openness, openness, out=self._max_openness)

    def reset(self, env_ids=None) -> None:
        env_index = slice(None) if env_ids is None else env_ids
        self._min_openness[env_index] = float("inf")
        self._max_openness[env_index] = float("-inf")

    def __call__(self, env, env_id: int, openable_object) -> dict[str, Any] | None:
        # An episode that ends before its first step still holds both starting values. Any step moves at
        # least one of them, including a NaN or infinite openness, so this is not mistaken for a bad value.
        has_no_steps = (self._min_openness[env_id] == float("inf")) & (self._max_openness[env_id] == float("-inf"))
        if has_no_steps:
            return None
        return {
            "object": openable_object.name,
            "min_openness": _finite_or_none(self._min_openness[env_id]),
            "max_openness": _finite_or_none(self._max_openness[env_id]),
        }
