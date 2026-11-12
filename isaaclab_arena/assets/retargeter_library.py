# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Retargeters that map each teleoperation device and embodiment pair to a device configuration."""

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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_retargeter

if TYPE_CHECKING:
    from isaaclab.devices import DeviceCfg
    from isaaclab_teleop import IsaacTeleopCfg

    from isaaclab_arena.assets.device_library import TeleopDeviceBase


class RetargetterBase(ABC):
    """Base class for teleop retargeter entries in the Arena registry.

    Subclasses associate a (device, embodiment) pair with a pipeline builder
    function compatible with ``IsaacTeleopCfg.pipeline_builder``.
    """

    device: str
    embodiment: str

    @abstractmethod
    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return an isaacteleop pipeline builder callable, or None if not applicable.

        Args:
            embodiment: Embodiment the pipeline is built for.
        """
        raise NotImplementedError

    def get_device_cfg(self, device: TeleopDeviceBase, embodiment: object) -> DeviceCfg | IsaacTeleopCfg:
        """Configure a device to emit this embodiment's actions.

        Args:
            device: Arena teleoperation device to configure.
            embodiment: Embodiment receiving the device's actions.

        Returns:
            Configuration for the selected device and embodiment.
        """
        return device.get_device_cfg(pipeline_builder=self.get_pipeline_builder(embodiment), embodiment=embodiment)


@register_retargeter
class GR1T2PinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for GR1T2 with Pink IK and dex hand retargeting."""

    device = "openxr"
    embodiment = "gr1_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        """Return the GR1T2 pick-and-place pipeline builder from Isaac Lab.

        Args:
            embodiment: Embodiment the pipeline is built for.
        """
        from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import build_gr1t2_pickplace_pipeline

        return lambda: build_gr1t2_pickplace_pipeline()[0]


@register_retargeter
class G1WbcPinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for G1 WBC Pink (locomanipulation: wrist + TriHand + locomotion)."""

    device = "openxr"
    embodiment = "g1_wbc_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        """Return the G1 Pink locomanipulation pipeline builder.

        Args:
            embodiment: Embodiment the pipeline is built for.
        """
        from isaaclab_arena_g1.teleop.g1_pink_locomanipulation_pipeline import _build_g1_pink_locomanipulation_pipeline

        return _build_g1_pink_locomanipulation_pipeline


@register_retargeter
class G1WbcAgilePinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for G1 WBC AGILE with Pink IK."""

    device = "openxr"
    embodiment = "g1_wbc_agile_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        """Return the G1 Pink locomanipulation pipeline builder.

        Args:
            embodiment: Embodiment the pipeline is built for.
        """
        from isaaclab_arena_g1.teleop.g1_pink_locomanipulation_pipeline import _build_g1_pink_locomanipulation_pipeline

        return _build_g1_pink_locomanipulation_pipeline


@register_retargeter
class FrankaKeyboardRetargeter(RetargetterBase):
    """Keyboard retargeter for the Franka embodiment with IK control, which needs no pipeline."""

    device = "keyboard"
    embodiment = "franka_ik"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return None, since the keyboard needs no isaacteleop pipeline.

        Args:
            embodiment: Embodiment the pipeline would be built for.
        """
        return None  # noqa: R501


@register_retargeter
class FrankaSpaceMouseRetargeter(RetargetterBase):
    """SpaceMouse retargeter for the Franka embodiment with IK control, which needs no pipeline."""

    device = "spacemouse"
    embodiment = "franka_ik"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return None, since the SpaceMouse needs no isaacteleop pipeline.

        Args:
            embodiment: Embodiment the pipeline would be built for.
        """
        return None  # noqa: R501


@register_retargeter
class DroidDifferentialIKKeyboardRetargeter(RetargetterBase):
    """Keyboard retargeter for the DROID embodiment with differential IK control."""

    device = "keyboard"
    embodiment = "droid_differential_ik"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return None, since the keyboard needs no isaacteleop pipeline.

        Args:
            embodiment: Embodiment the pipeline would be built for.
        """
        return None  # noqa: R501

    def get_device_cfg(self, device: TeleopDeviceBase, embodiment: object) -> DeviceCfg:
        """Configure the keyboard with the DROID keyboard device class.

        Args:
            device: Arena keyboard device to configure.
            embodiment: Embodiment receiving the device's actions.
        """
        from isaaclab_arena.embodiments.droid.teleop import DroidSe3Keyboard

        cfg = super().get_device_cfg(device, embodiment)
        cfg.class_type = DroidSe3Keyboard
        return cfg


@register_retargeter
class DroidDifferentialIKSpaceMouseRetargeter(RetargetterBase):
    """SpaceMouse retargeter for the DROID embodiment with differential IK control."""

    device = "spacemouse"
    embodiment = "droid_differential_ik"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return None, since the SpaceMouse needs no isaacteleop pipeline.

        Args:
            embodiment: Embodiment the pipeline would be built for.
        """
        return None  # noqa: R501

    def get_device_cfg(self, device: TeleopDeviceBase, embodiment: object) -> DeviceCfg:
        """Configure the SpaceMouse with the DROID SpaceMouse device class.

        Args:
            device: Arena SpaceMouse device to configure.
            embodiment: Embodiment receiving the device's actions.
        """
        from isaaclab_arena.embodiments.droid.teleop import DroidSe3SpaceMouse

        cfg = super().get_device_cfg(device, embodiment)
        cfg.class_type = DroidSe3SpaceMouse
        return cfg


@register_retargeter
class AgibotKeyboardRetargeter(RetargetterBase):
    """Keyboard retargeter for the Agibot embodiment, which needs no pipeline."""

    device = "keyboard"
    embodiment = "agibot"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return None, since the keyboard needs no isaacteleop pipeline.

        Args:
            embodiment: Embodiment the pipeline would be built for.
        """
        return None  # noqa: R501


@register_retargeter
class GalbotKeyboardRetargeter(RetargetterBase):
    """Keyboard retargeter for the Galbot embodiment, which needs no pipeline."""

    device = "keyboard"
    embodiment = "galbot"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return None, since the keyboard needs no isaacteleop pipeline.

        Args:
            embodiment: Embodiment the pipeline would be built for.
        """
        return None  # noqa: R501
