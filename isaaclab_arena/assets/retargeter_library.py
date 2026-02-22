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

from isaaclab.devices.openxr.retargeters import GR1T2RetargeterCfg
from isaaclab.devices.retargeter_base import RetargeterCfg

from isaaclab_arena.assets.register import register_retargeter


class RetargetterBase(ABC):
    device: str
    embodiment: str

    @abstractmethod
    def get_retargeter_cfg(
        self, embodiment: object, sim_device: str, enable_visualization: bool = False
    ) -> RetargeterCfg:
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
class DroidDifferentialIKKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "droid_differential_ik"

    def __init__(self):
        pass

    def get_retargeter_cfg(
        self, droid_embodiment, sim_device: str, enable_visualization: bool = False
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
class GalbotKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "galbot"

    def __init__(self):
        pass

    def get_retargeter_cfg(
        self, galbot_embodiment, sim_device: str, enable_visualization: bool = False
    ) -> RetargeterCfg | None:
        return None
