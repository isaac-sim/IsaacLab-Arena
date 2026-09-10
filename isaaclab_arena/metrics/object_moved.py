# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


class ObjectVelocityRecorder(RecorderTerm):
    """Records the average speed of an object for each simulation step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Get name from config instead of class attribute
        self.name = cfg.name
        self.object_name = cfg.object_name

    def record_post_step(self):
        object_average_speed = self._env.arena_world.get_average_speed_w(self.object_name).unsqueeze(-1)
        assert object_average_speed.shape == (self._env.num_envs, 1)
        return self.name, object_average_speed


@configclass
class ObjectVelocityRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ObjectVelocityRecorder
    name: str = "object_linear_velocity"
    object_name: str = MISSING


def compute_object_moved_rate(recorded_metric_data: list[np.ndarray], object_velocity_threshold: float) -> float:
    """Computes the object-moved rate from the recorded metric data.

    Args:
        recorded_metric_data(list[np.ndarray]): The recorded object velocity per simulated episode.
        object_velocity_threshold(float): The threshold for the object velocity to be considered moved.

    Returns:
        The object-moved rate(float). Value between 0 and 1. The proportion of episodes
            in which the object moved.
    """
    object_velocity_per_demo = recorded_metric_data
    object_moved_per_demo = []
    for object_velocity in object_velocity_per_demo:
        assert object_velocity.ndim == 2
        assert object_velocity.shape[1] in (1, 3)
        object_linear_velocity_magnitude = np.linalg.norm(object_velocity, axis=-1)
        object_moved = np.any(object_linear_velocity_magnitude > object_velocity_threshold)
        object_moved_per_demo.append(object_moved)
    object_moved_rate = np.mean(object_moved_per_demo)
    return object_moved_rate


class ObjectMovedRateMetric(MetricBase):
    """Computes the object-moved rate.

    The object-moved rate is the number of episodes in which the object moved, divided
    by the total number of episodes.
    """

    name = "object_moved_rate"
    recorder_term_name = "object_linear_velocity"

    def __init__(self, object: Asset, object_velocity_threshold: float = 0.5):
        """Initializes the object-moved rate metric.

        Args:
            object(Asset): The object to compute the object-moved rate for.
            object_velocity_threshold(float): The threshold for the object velocity to be considered moved.
        """
        super().__init__()
        self.object = object
        self.object_velocity_threshold = object_velocity_threshold

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the object-moved rate metric."""
        return ObjectVelocityRecorderCfg(name=self.recorder_term_name, object_name=self.object.name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        """Return the metric term configuration for the object-moved rate metric."""
        return MetricTermCfg(
            compute_metric_func=compute_object_moved_rate,
            params={"object_velocity_threshold": self.object_velocity_threshold},
            recorder_term_name=self.recorder_term_name,
        )
