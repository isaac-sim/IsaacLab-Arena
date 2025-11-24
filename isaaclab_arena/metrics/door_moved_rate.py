# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.metrics.metric_base import MetricBase


class OpennessRecorder(RecorderTerm):
    """Records the openness of an object for each sim step of an episode."""

    name = "openness"

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.object = cfg.object

    def record_post_step(self):
        openness = self.object.get_openness(self._env)
        return self.name, openness


@configclass
class JointStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = OpennessRecorder
    object: ObjectBase = MISSING


class DoorMovedRateMetric(MetricBase):
    """Computes the door-moved rate.

    The door-moved rate is the number of episodes in which the door moved, divided
    by the total number of episodes.
    """

    name = "door_moved_rate"
    recorder_term_name = OpennessRecorder.name

    def __init__(self, object: Openable, reset_openness: float, openness_delta_threshold: float = 0.05):
        """Initializes the door-moved rate metric.

        Args:
            object(Openable): The door to compute the door-moved rate for.
            reset_openness(float): The initial openness of the door (what the door resets to).
            openness_delta_threshold(float): The threshold for the door openness to be considered
                moved. This is relative to the initial openness of the door.
        """
        super().__init__()
        assert isinstance(object, Openable), "Object must be Openable"
        self.object = object
        self.reset_openness = reset_openness
        self.openness_delta_threshold = openness_delta_threshold

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the door-moved rate metric."""
        return JointStateRecorderCfg(object=self.object)

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        """Computes the door-moved rate from the recorded metric data.

        Args:
            recorded_metric_data(list[np.ndarray]): The recorded door openness per simulated
                episode.

        Returns:
            The door-moved rate(float). Value between 0 and 1. The proportion of episodes
                in which the door moved.
        """
        if len(recorded_metric_data) == 0:
            return 0.0
        door_moved_per_demo = []
        for episode_data in recorded_metric_data:
            openness_threshold = self.reset_openness + self.openness_delta_threshold
            door_moved_per_demo.append(np.any(episode_data > openness_threshold))
        door_moved_rate = np.mean(door_moved_per_demo)
        return door_moved_rate
