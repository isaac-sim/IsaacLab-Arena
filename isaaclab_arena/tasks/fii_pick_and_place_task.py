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

import numpy as np
from dataclasses import MISSING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab.envs.common import ViewerCfg

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.common import ViewerCfg

from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from dataclasses import MISSING

from isaaclab.utils import configclass

class FiiPickAndPlaceTask(TaskBase):

    def __init__(
        self,
        pick_up_object: Asset,
        packing_table: Asset,
        episode_length_s: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.scene_config = FiibotSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        self.termination_cfg = FiibotTerminationsCfg()
        self.viewer_cfg = ViewerCfg(
        eye=(0.0, 3.0, 1.5), lookat=(0.0, 0.0, 0.7), origin_type="asset_body", asset_name="robot", body_name="base_link"
    )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        pass

    def get_viewer_cfg(self) -> ViewerCfg:
        return self.viewer_cfg
    def get_metrics(self):
        pass
    def get_prompt(self):
        pass
    def get_mimic_env_cfg(self, embodiment_name: str):
        pass
#=======================================================================
#   TERMINATIONS
#=======================================================================
@configclass
class FiibotTerminationsCfg:
    
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=manip_mdp.task_done_pick_place, params={
        "object_cfg": SceneEntityCfg("object"),
        "task_link_name": "right_7_Link",
        "right_wrist_max_x": 0.26,
        "min_x": 0.40,
        "max_x": 0.85,
        "min_y": 0.35,
        "max_y": 0.8, 
        "max_height": 1.10,
        "min_vel": 0.20,                                                        
    })

#=======================================================================
#   SCENE
#=======================================================================

class FiibotSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0, 
            color=(0.75, 0.75, 0.75)
        )
    )