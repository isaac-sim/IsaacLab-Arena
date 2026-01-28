# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.devices.openxr.retargeters import (
    G1LowerBodyStandingMotionControllerRetargeterCfg,
    G1TriHandUpperBodyMotionControllerGripperRetargeterCfg,
    GR1T2RetargeterCfg,
)
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

from isaaclab_arena.assets.register import register_retargeter


class DummyTorsoRetargeter(RetargeterBase):
    """Dummy retargeter that returns zero torso orientation commands.
    
    This is used to pad the action space for G1 WBC Pink with motion controllers,
    which don't provide torso orientation commands.
    """

    def __init__(self, cfg: "DummyTorsoRetargeterCfg"):
        super().__init__(cfg)

    def retarget(self, data: Any) -> torch.Tensor:
        """Return zeros for torso orientation (roll, pitch, yaw)."""
        return torch.zeros(3, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """This retargeter doesn't require any device data."""
        return []


@dataclass
class DummyTorsoRetargeterCfg(RetargeterCfg):
    """Configuration for dummy torso retargeter."""

    retargeter_type: type[RetargeterBase] = DummyTorsoRetargeter


class RetargetterBase(ABC):
    device: str
    embodiment: str

    @abstractmethod
    def get_retargeter_cfg(
        self, embodiment: object, sim_device: str, enable_visualization: bool = False
    ) -> RetargeterCfg | list[RetargeterCfg] | None:
        """Get retargeter configuration.
        
        Can return:
        - A single RetargeterCfg
        - A list of RetargeterCfg (for devices needing multiple retargeters)
        - None (for devices that don't need retargeters)
        """
        raise NotImplementedError


@register_retargeter
class GR1T2PinkOpenXRRetargeter(RetargetterBase):

    device = "openxr"
    embodiment = "gr1_pink"
    num_open_xr_hand_joints = 52

    def __init__(self):
        pass

    def get_retargeter_cfg(
        self, gr1t2_embodiment, sim_device: str, enable_visualization: bool = False
    ) -> RetargeterCfg:
        return GR1T2RetargeterCfg(
            enable_visualization=enable_visualization,
            # number of joints in both hands
            num_open_xr_hand_joints=self.num_open_xr_hand_joints,
            sim_device=sim_device,
            hand_joint_names=gr1t2_embodiment.get_action_cfg().upper_body_ik.hand_joint_names,
        )


@register_retargeter
class FrankaKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "franka"

    def __init__(self):
        pass

    def get_retargeter_cfg(
        self, franka_embodiment, sim_device: str, enable_visualization: bool = False
    ) -> RetargeterCfg | None:
        return None


@register_retargeter
class FrankaSpaceMouseRetargeter(RetargetterBase):
    device = "spacemouse"
    embodiment = "franka"

    def __init__(self):
        pass

    def get_retargeter_cfg(
        self, franka_embodiment, sim_device: str, enable_visualization: bool = False
    ) -> RetargeterCfg | None:
        return None


@register_retargeter
class AgibotKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "agibot"

    def __init__(self):
        pass

    def get_retargeter_cfg(
        self, agibot_embodiment, sim_device: str, enable_visualization: bool = False
    ) -> RetargeterCfg | None:
        return None


@register_retargeter
class G1WbcPinkMotionControllersRetargeter(RetargetterBase):
    """Retargeter for G1 WBC Pink embodiment with motion controllers (Quest controllers)."""

    device = "motion_controllers"
    embodiment = "g1_wbc_pink"

    def __init__(self):
        pass

    def get_retargeter_cfg(
        self, g1_embodiment, sim_device: str, enable_visualization: bool = False
    ) -> list[RetargeterCfg]:
        """Get motion controller retargeter configuration for G1.
        
        Returns a list of retargeters:
        - Upper body (with gripper): outputs 16 dims [gripper(2), left_wrist(7), right_wrist(7)]
        - Lower body: outputs 4 dims [nav_cmd(3), hip_height(1)]
        - Dummy torso: outputs 3 dims [torso_orientation_rpy(3)] all zeros
        Total: 23 dims to match g1_wbc_pink action space
        """
        return [
            G1TriHandUpperBodyMotionControllerGripperRetargeterCfg(sim_device=sim_device),
            G1LowerBodyStandingMotionControllerRetargeterCfg(sim_device=sim_device),
            DummyTorsoRetargeterCfg(sim_device=sim_device),
        ]
