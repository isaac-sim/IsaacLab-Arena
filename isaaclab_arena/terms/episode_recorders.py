# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode recorder terms that tasks and environments can add to record a value per episode."""

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


class ObjectDisplacement(EpisodeRecorderTerm):
    """Records the largest distance [m] an object got from where it was after the episode's first step.

    The start position is taken after the episode's first control step, so motion during that step, such as a
    short drop onto the support after spawning, does not count; settling that lasts longer does. An object that
    is moved and put back still counts with its largest distance. Deformable objects use their mean node position.
    """

    def __init__(self, cfg: EpisodeRecorderTermCfg, env):
        super().__init__(cfg, env)
        self._object_name = cfg.params["object_name"]
        self._start_position_w = torch.zeros(env.num_envs, 3, device=env.device)
        self._has_start_position = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._largest_displacement = torch.zeros(env.num_envs, device=env.device)

    def post_step(self):
        position_w = self._env.arena_world.get_position_w(self._object_name)
        # Update in place, for the same reason as in ObjectPeakSpeed.post_step().
        needs_start_position = ~self._has_start_position
        self._start_position_w[needs_start_position] = position_w[needs_start_position]
        self._has_start_position.fill_(True)
        displacement = torch.linalg.norm(position_w - self._start_position_w, dim=-1)
        torch.maximum(self._largest_displacement, displacement, out=self._largest_displacement)

    def reset(self, env_ids=None):
        selected_envs = slice(None) if env_ids is None else env_ids
        self._has_start_position[selected_envs] = False
        self._largest_displacement[selected_envs] = 0.0

    def __call__(self, env, env_id: int, object_name: str) -> float:
        # The manager passes every cfg.params entry as a keyword argument, so object_name must be
        # accepted here even though __init__ already read it.
        return float(self._largest_displacement[env_id])


class RevoluteJointMovement(EpisodeRecorderTerm):
    """Records how far a revolute joint moved from its reset openness during the episode.

    Openness is the joint position as a fraction of its range, from 0 (closed) to 1 (open), so the movement is
    a fraction of the range too. A joint that is moved and moved back still counts with its largest movement.
    """

    def __init__(self, cfg: EpisodeRecorderTermCfg, env):
        super().__init__(cfg, env)
        self._openable_object = cfg.params["openable_object"]
        self._reset_openness = cfg.params["reset_openness"]
        self._largest_movement = torch.zeros(env.num_envs, device=env.device)

    def post_step(self):
        openness = self._openable_object.get_openness(self._env)
        movement = torch.abs(openness - self._reset_openness)
        # Update in place, for the same reason as in ObjectPeakSpeed.post_step().
        torch.maximum(self._largest_movement, movement, out=self._largest_movement)

    def reset(self, env_ids=None):
        self._largest_movement[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, env_id: int, openable_object, reset_openness: float) -> float:
        # The manager passes every cfg.params entry as a keyword argument, so both must be accepted here.
        return float(self._largest_movement[env_id])
