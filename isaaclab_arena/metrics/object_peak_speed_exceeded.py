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
    """Records mean world-frame linear velocity of an object for each simulation step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Get name from config instead of class attribute
        self.name = cfg.name
        self.object_name = cfg.object_name

    def record_post_step(self):
        # Use mean linear velocity for overall object movement. This is defined for both rigidbodies and deformables.
        object_linear_velocity_w = self._env.arena_world.get_mean_linear_velocity_w(self.object_name)
        assert object_linear_velocity_w.shape == (self._env.num_envs, 3)
        return self.name, object_linear_velocity_w


@configclass
class ObjectVelocityRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ObjectVelocityRecorder
    name: str = "object_linear_velocity"
    object_name: str = MISSING


def compute_object_peak_speed_exceeded_rate(
    recorded_metric_data: list[np.ndarray], object_velocity_threshold: float
) -> float:
    """Computes the fraction of episodes in which the object's peak speed exceeds a threshold.

    Args:
        recorded_metric_data(list[np.ndarray]): The recorded object velocity per simulated episode.
        object_velocity_threshold(float): The speed (m/s) above which an episode counts as exceeded.

    Returns:
        The peak-speed-exceeded rate(float). Value between 0 and 1. A slow movement that never
            exceeds the threshold is not counted.
    """
    object_velocity_per_demo = recorded_metric_data
    peak_speed_exceeded_per_demo = []
    for object_velocity in object_velocity_per_demo:
        assert object_velocity.ndim == 2
        assert object_velocity.shape[1] == 3
        object_linear_velocity_magnitude = np.linalg.norm(object_velocity, axis=-1)
        peak_speed_exceeded = np.any(object_linear_velocity_magnitude > object_velocity_threshold)
        peak_speed_exceeded_per_demo.append(peak_speed_exceeded)
    object_peak_speed_exceeded_rate = np.mean(peak_speed_exceeded_per_demo)
    return object_peak_speed_exceeded_rate


class ObjectPeakSpeedExceededRateMetric(MetricBase):
    """Computes the object peak-speed-exceeded rate.

    The rate is the number of episodes in which the object's linear speed exceeded the threshold
    at any step, divided by the total number of episodes. It measures peak speed, not displacement.
    """

    name = "object_peak_speed_exceeded_rate"
    recorder_term_name = "object_linear_velocity"

    def __init__(self, object: Asset, object_velocity_threshold: float = 0.5):
        """Initializes the object peak-speed-exceeded rate metric.

        Args:
            object(Asset): The object to compute the peak-speed-exceeded rate for.
            object_velocity_threshold(float): The speed (m/s) above which an episode counts as exceeded.
        """
        super().__init__()
        self.object = object
        self.object_velocity_threshold = object_velocity_threshold

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the object peak-speed-exceeded rate metric."""
        return ObjectVelocityRecorderCfg(name=self.recorder_term_name, object_name=self.object.name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        """Return the metric term configuration for the object peak-speed-exceeded rate metric."""
        return MetricTermCfg(
            compute_metric_func=compute_object_peak_speed_exceeded_rate,
            params={"object_velocity_threshold": self.object_velocity_threshold},
            recorder_term_name=self.recorder_term_name,
        )
