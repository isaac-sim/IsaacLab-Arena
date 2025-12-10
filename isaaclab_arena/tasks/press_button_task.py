# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class PressButtonTask(TaskBase):
    def __init__(
        self,
        pressable_object: Pressable,
        pressedness_threshold: float | None = None,
        reset_pressedness: float | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(pressable_object, Pressable), "Pressable object must be an instance of Pressable"
        self.pressable_object = pressable_object
        self.pressedness_threshold = pressedness_threshold
        self.reset_pressedness = reset_pressedness
        self.task_description = (
            f"Press the {pressable_object.name} button" if task_description is None else task_description
        )

    def get_scene_cfg(self):
        pass

    def get_termination_cfg(self):
        params = {}
        if self.pressedness_threshold is not None:
            params["threshold"] = self.pressedness_threshold
        success = TerminationTermCfg(
            func=self.pressable_object.is_pressed,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return PressEventCfg(self.pressable_object, reset_pressedness=self.reset_pressedness)

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pressable_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING


@configclass
class PressEventCfg:
    """Configuration for Open Door."""

    reset_button_state: EventTermCfg = MISSING

    def __init__(self, pressable_object: Pressable, reset_pressedness: float | None):
        assert isinstance(pressable_object, Pressable), "Object pose must be an instance of Pressable"
        params = {}
        if reset_pressedness is not None:
            params["unpressed_percentage"] = reset_pressedness
        self.reset_button_state = EventTermCfg(
            func=pressable_object.unpress,
            mode="reset",
            params=params,
        )
