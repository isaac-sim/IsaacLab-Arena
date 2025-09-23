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

from isaac_arena.affordances.openable import Openable
from isaac_arena.tasks.task import TaskBase


class OpenDoorTask(TaskBase):
    def __init__(
        self, openable_object: Openable, openness_threshold: float | None = None, reset_openness: float | None = None
    ):
        super().__init__()
        assert isinstance(openable_object, Openable), "Openable object must be an instance of Openable"
        self.openable_object = openable_object
        self.openness_threshold = openness_threshold
        self.reset_openness = reset_openness

    def get_scene_cfg(self):
        pass

    def get_termination_cfg(self):
        params = {}
        if self.openness_threshold is not None:
            params["threshold"] = self.openness_threshold
        success = TerminationTermCfg(
            func=self.openable_object.is_open,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return OpenDoorEventCfg(self.openable_object, reset_openness=self.reset_openness)

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=False)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING


@configclass
class OpenDoorEventCfg:
    """Configuration for Open Door."""

    reset_door_state: EventTermCfg = MISSING

    def __init__(self, openable_object: Openable, reset_openness: float | None):
        assert isinstance(openable_object, Openable), "Object pose must be an instance of Openable"
        params = {}
        if reset_openness is not None:
            params["percentage"] = reset_openness
        self.reset_door_state = EventTermCfg(
            func=openable_object.close,
            mode="reset",
            params=params,
        )
