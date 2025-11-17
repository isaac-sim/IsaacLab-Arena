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
from isaaclab_arena.tasks.terminations_orca import multi_stage_with_fixed_destination
from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask

class G1OrcaTask(G1LocomanipPickAndPlaceTask):
    def get_termination_cfg(self):
        # Two-stage success: 
        # Stage 1: Box on cart
        # Stage 2: Cart to target position (surgical bed)
        success = TerminationTermCfg(
            func=multi_stage_with_fixed_destination,
            params={
                # Stage 1: Box on cart
                "box_cfg": SceneEntityCfg(self.pick_up_object.name),
                "cart_cfg": SceneEntityCfg(self.destination_bin.name),
                "box_to_cart_max_x": 0.30,  # 30cm tolerance in X
                "box_to_cart_max_y": 0.20,  # 20cm tolerance in Y
                "box_to_cart_max_z": 0.92,  # 25cm tolerance in Z (box above cart)
                # Stage 2: Cart to fixed target position (surgical bed)
                "target_position_x": 0.137,
                "target_position_y": -4.5,
                "target_position_z": -0.7875,
                "cart_to_target_max_x": 0.30,  # 50cm tolerance in X
                "cart_to_target_max_y": 0.30,  # 50cm tolerance in Y
                "cart_to_target_max_z": 0.20,  # 20cm tolerance in Z
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": -0.6,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING
