# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaac_arena.affordances.pressable import Pressable
from isaac_arena.tasks.task import TaskBase


class PressButtonTask(TaskBase):
    def __init__(
        self, pressable_object: Pressable, pressed_threshold: float | None = None, reset_pressed: float | None = None
    ):
        super().__init__()
        assert isinstance(pressable_object, Pressable), "Pressable object must be an instance of Pressable"
        self.pressable_object = pressable_object
        self.pressed_threshold = pressed_threshold
        self.reset_pressed = reset_pressed

    def get_scene_cfg(self):
        pass

    def get_termination_cfg(self):
        params = {}
        if self.pressed_threshold is not None:
            params["threshold"] = self.pressed_threshold
        success = TerminationTermCfg(
            func=self.pressable_object.is_pressed,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return PressEventCfg(self.pressable_object, reset_pressed=self.reset_pressed)

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=False)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING


@configclass
class PressEventCfg:
    """Configuration for Open Door."""

    reset_button_state: EventTermCfg = MISSING

    def __init__(self, pressable_object: Pressable, reset_pressed: float | None):
        assert isinstance(pressable_object, Pressable), "Object pose must be an instance of Pressable"
        params = {}
        if reset_pressed is not None:
            params["unpressed_percentage"] = reset_pressed
        self.reset_button_state = EventTermCfg(
            func=pressable_object.unpress,
            mode="reset",
            params=params,
        )
