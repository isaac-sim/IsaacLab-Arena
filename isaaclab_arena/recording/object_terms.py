# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode recorder terms that follow an object during the episode."""

from __future__ import annotations

import torch

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg


class ObjectPeakSpeed(EpisodeRecorderTerm):
    """Records the highest linear speed [m/s] an object reached during the episode."""

    def __init__(self, cfg: EpisodeRecorderTermCfg, env):
        super().__init__(cfg, env)
        self._object_name = cfg.params["object_name"]
        self._peak_speed = torch.zeros(env.num_envs, device=env.device)

    def post_step(self):
        velocity_w = self._env.arena_world.get_mean_linear_velocity_w(self._object_name)
        speed = torch.linalg.norm(velocity_w, dim=-1)
        # Update in place: a tensor created inside torch.inference_mode(), where the env may be stepped,
        # cannot be written in place by reset() outside it.
        torch.maximum(self._peak_speed, speed, out=self._peak_speed)

    def reset(self, env_ids=None):
        self._peak_speed[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, env_id: int, object_name: str) -> float:
        # The manager passes every cfg.params entry as a keyword argument, so object_name must be
        # accepted here even though __init__ already read it.
        return float(self._peak_speed[env_id])
