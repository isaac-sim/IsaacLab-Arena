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
from collections.abc import Callable

from isaaclab_arena.assets.register import register_retargeter


class RetargetterBase(ABC):
    """Base class for teleop retargeter entries in the Arena registry.

    Subclasses associate a (device, embodiment) pair with a pipeline builder
    function compatible with ``IsaacTeleopCfg.pipeline_builder``.
    """

    device: str
    embodiment: str

    @abstractmethod
    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return an isaacteleop pipeline builder callable, or None if not applicable."""
        raise NotImplementedError


@register_retargeter
class GR1T2PinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for GR1T2 with Pink IK and dex hand retargeting."""

    device = "openxr"
    embodiment = "gr1_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
            _build_gr1t2_pickplace_pipeline,
        )

        return lambda: _build_gr1t2_pickplace_pipeline()[0]


@register_retargeter
class G1WbcPinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for G1 WBC Pink (locomanipulation: wrist + TriHand + locomotion)."""

    device = "openxr"
    embodiment = "g1_wbc_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        from isaaclab_arena.assets.g1_pink_locomanipulation_pipeline import _build_g1_pink_locomanipulation_pipeline

        return _build_g1_pink_locomanipulation_pipeline


@register_retargeter
class FrankaKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "franka"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class FrankaSpaceMouseRetargeter(RetargetterBase):
    device = "spacemouse"
    embodiment = "franka"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class DroidDifferentialIKKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "droid_differential_ik"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


# @register_retargeter
# class AgibotKeyboardRetargeter(RetargetterBase):
#     device = "keyboard"
#     embodiment = "agibot"

#     def __init__(self):
#         pass

#     def get_pipeline_builder(self, embodiment: object) -> Callable | None:
#         return None


@register_retargeter
class GalbotKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "galbot"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None
