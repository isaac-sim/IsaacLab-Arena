# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_base import MetricBase


class TaskBase(ABC):

    def __init__(self, episode_length_s: float | None = None, task_description: str | None = None):
        self.episode_length_s = episode_length_s
        self.task_description = task_description

    @abstractmethod
    def get_scene_cfg(self) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_termination_cfg(self) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_events_cfg(self) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_metrics(self) -> list[MetricBase]:
        raise NotImplementedError("Function not implemented yet.")

    def get_observation_cfg(self) -> Any:
        return None

    def get_rewards_cfg(self) -> Any:
        return None

    def get_curriculum_cfg(self) -> Any:
        return None

    def get_commands_cfg(self) -> Any:
        return None

    def get_recorder_term_cfg(self) -> RecorderManagerBaseCfg:
        return None

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg()

    def get_episode_length_s(self) -> float | None:
        return self.episode_length_s

    def get_task_description(self) -> str | None:
        return self.task_description
