# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tasks.task_base import TaskBase


class GenericTask(TaskBase):
    """Generic task wrapper for deserialized task data."""

    def __init__(self, scene_cfg, events_cfg, termination_cfg,
                    observation_cfg, rewards_cfg, curriculum_cfg, commands_cfg,
                    episode_length_s=None):
        super().__init__(episode_length_s=episode_length_s)

        # Store deserialized config attributes
        self.scene_config = scene_cfg
        self.events_cfg_data = events_cfg
        self.termination_cfg_data = termination_cfg
        self.observation_cfg_data = observation_cfg
        self.rewards_cfg_data = rewards_cfg
        self.curriculum_cfg_data = curriculum_cfg
        self.commands_cfg_data = commands_cfg

    def get_scene_cfg(self):
        """Returns task-specific scene config if available."""
        return self.scene_config

    def get_termination_cfg(self):
        """Returns task-specific termination config."""
        return self.termination_cfg_data

    def get_events_cfg(self):
        """Returns task-specific events config."""
        return self.events_cfg_data

    def get_observation_cfg(self):
        """Returns task-specific observation config."""
        return self.observation_cfg_data

    def get_rewards_cfg(self):
        """Returns task-specific rewards config."""
        return self.rewards_cfg_data

    def get_curriculum_cfg(self):
        """Returns task-specific curriculum config."""
        return self.curriculum_cfg_data

    def get_commands_cfg(self):
        """Returns task-specific commands config."""
        return self.commands_cfg_data

    def get_prompt(self):
        """Returns task prompt if available."""
        return self.task_data.get('prompt', '')

    def get_mimic_env_cfg(self, embodiment_name: str):
        """Returns mimic env config (not available in generic task)."""
        return None

    def get_metrics(self):
        """Returns empty metrics list (metrics are stored at cfg level)."""
        return []
