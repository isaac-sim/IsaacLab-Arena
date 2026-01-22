# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase


class LiftSuccessRecorder(RecorderTerm):
    """Records whether the object was successfully lifted during an episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.minimum_height = cfg.params["minimum_height"]
        self.object_cfg = cfg.params["object_cfg"]
        self.first_reset = True

    def record_pre_reset(self, env_ids):
        """Record if object was lifted above minimum height before reset."""
        # Skip first reset
        if self.first_reset:
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            return None, None

        # Get the object
        object: RigidObject = self._env.scene[self.object_cfg.name]
        
        # Check if object is above minimum height
        object_height = object.data.root_pos_w[env_ids, 2]
        lift_success = object_height > self.minimum_height

        return self.name, lift_success


@configclass
class LiftSuccessRecorderCfg(RecorderTermCfg):
    """Configuration for the lift success recorder."""

    class_type: type[RecorderTerm] = LiftSuccessRecorder
    name: str = "lift_success"
    params: dict = {}  # Will be set by metric


class LiftSuccessMetric(MetricBase):
    """
    Computes the lift success rate.
    
    The lift success rate is the proportion of episodes where the object
    was lifted above the minimum height threshold.
    """

    name = "lift_success_rate"
    recorder_term_name = "lift_success"

    def __init__(self, minimum_height: float, object_name: str = "object"):
        """
        Initialize the lift success metric.
        
        Args:
            minimum_height: Minimum height threshold for successful lift (in meters)
            object_name: Name of the object asset in the scene
        """
        self.minimum_height = minimum_height
        self.object_name = object_name

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for lift success."""
        return LiftSuccessRecorderCfg(
            name=self.recorder_term_name,
            params={
                "minimum_height": self.minimum_height,
                "object_cfg": SceneEntityCfg(self.object_name),
            },
        )

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        """
        Compute lift success rate from recorded data.
        
        Args:
            recorded_metric_data: List of boolean arrays indicating lift success per episode
            
        Returns:
            Success rate as a float between 0 and 1
        """
        num_episodes = len(recorded_metric_data)
        if num_episodes == 0:
            return 0.0

        all_episodes_success = np.concatenate(recorded_metric_data)
        success_rate = np.mean(all_episodes_success)
        return float(success_rate)
