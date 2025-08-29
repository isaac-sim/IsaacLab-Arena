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

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaac_arena.tasks.task import TaskBase


class OpenDoorTask(TaskBase):
    def __init__(self):
        super().__init__()

    def get_termination_cfg(self):
        return TerminationsCfg()

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=False)

    # TODO(alexmillane, 2025.08.28): Implement success condition.
    # success = TerminationTermCfg(
    #     func=object_on_destination,
    #     params={
    #         "object_cfg": SceneEntityCfg("pick_up_object"),
    #         "contact_sensor_cfg": SceneEntityCfg("pick_up_object_contact_sensor"),
    #         "force_threshold": 1.0,
    #         "velocity_threshold": 0.1,
    #     },
    # )
