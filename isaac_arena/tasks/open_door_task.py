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
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaac_arena.assets.affordances import Openable
from isaac_arena.tasks.task import TaskBase


class OpenDoorTask(TaskBase):
    def __init__(self, openable_object: Openable):
        super().__init__()
        assert isinstance(openable_object, Openable), "Openable object must be an instance of Openable"
        self.openable_object = openable_object

    def get_termination_cfg(self):
        # TODO(alexmillane, 2025.08.29): This is strongly coupled to the OpenDoorScene,
        # because of our use of the name "interactable_object" which is the name of the
        # member of the scene where the openable object is located... Need to improve
        # this design...
        success = TerminationTermCfg(
            func=self.openable_object.is_open,
            params={
                "asset_cfg": SceneEntityCfg("interactable_object"),
            },
        )
        return TerminationsCfg(success=success)

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
