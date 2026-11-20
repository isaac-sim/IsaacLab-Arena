# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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


class ObjectPositionRecorder(RecorderTerm):
    """Records the world-frame position of an object for each simulation step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.object_name = cfg.object_name

    def record_post_step(self):
        # Deformables report their mean nodal position, rigid objects their root position.
        t_W = self._env.arena_world.get_position_w(self.object_name)
        return self.name, t_W


@configclass
class ObjectPositionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ObjectPositionRecorder
    name: str = "object_position"
    object_name: str = MISSING


def compute_object_displaced_rate(recorded_metric_data: list[np.ndarray], displacement_threshold_m: float) -> float:
    """Computes the fraction of episodes in which the object was ever displaced from its first recorded position.

    An episode counts once any recorded position is past the threshold, even if the object is later put back.

    Args:
        recorded_metric_data(list[np.ndarray]): The recorded object position (num_steps, 3) per episode.
        displacement_threshold_m(float): The distance (m) from the first recorded position above which
            the object counts as displaced.

    Returns:
        The object-displaced rate(float). Value between 0 and 1. Independent of how fast the object moved.
    """
    if len(recorded_metric_data) == 0:
        return 0.0
    displaced_per_episode = []
    for position_w in recorded_metric_data:
        assert position_w.ndim == 2
        assert position_w.shape[1] == 3
        distance_from_start = np.linalg.norm(position_w - position_w[0], axis=-1)
        displaced_per_episode.append(np.any(distance_from_start > displacement_threshold_m))
    return float(np.mean(displaced_per_episode))


class ObjectDisplacedRateMetric(MetricBase):
    """Computes the object-displaced rate.

    The rate is the number of episodes in which the object's position, at any step, was farther than
    the threshold from its first recorded position, divided by the total number of episodes. Unlike
    a speed-based metric, a slow carry counts as displaced.

    The first position is recorded after the first environment step, not at reset, so motion before
    that step, such as an object dropping onto its support after spawning, does not count toward the
    displacement.
    """

    name = "object_displaced_rate"
    recorder_term_name = "object_position"

    def __init__(self, object: Asset, displacement_threshold_m: float = 0.02):
        """Initializes the object-displaced rate metric.

        Args:
            object(Asset): The object to compute the displaced rate for.
            displacement_threshold_m(float): The distance (m) from the starting position above which the
                object counts as displaced.
        """
        super().__init__()
        self.object = object
        self.displacement_threshold_m = displacement_threshold_m

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the object-displaced rate metric."""
        return ObjectPositionRecorderCfg(name=self.recorder_term_name, object_name=self.object.name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        """Return the metric term configuration for the object-displaced rate metric."""
        return MetricTermCfg(
            compute_metric_func=compute_object_displaced_rate,
            params={"displacement_threshold_m": self.displacement_threshold_m},
            recorder_term_name=self.recorder_term_name,
        )
