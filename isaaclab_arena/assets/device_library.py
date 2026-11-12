# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Teleoperation devices registered with Arena."""

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

from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCfg

from isaaclab_arena.assets.register import register_device


class TeleopDeviceBase(ABC):
    """Base class for teleoperation devices that can be registered by name."""

    name: str | None = None

    def __init__(self, sim_device: str | None = None):
        """Create a teleoperation device.

        Args:
            sim_device: Device the simulation runs on. None leaves it unspecified.
        """
        self.sim_device = sim_device

    @abstractmethod
    def get_device_cfg(self, pipeline_builder: Callable | None = None, embodiment: object | None = None):
        """Return the Isaac Lab configuration of this device.

        Args:
            pipeline_builder: Isaac Teleop pipeline builder for devices that use one.
            embodiment: Embodiment the device controls.
        """
        raise NotImplementedError


@register_device
class OpenXRCfg(TeleopDeviceBase):
    """OpenXR device driven through an Isaac Teleop pipeline."""

    name = "openxr"

    def __init__(self, sim_device: str | None = None):
        """Create an OpenXR device.

        Args:
            sim_device: Device the simulation runs on. None leaves it unspecified.
        """
        super().__init__(sim_device=sim_device)

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> IsaacTeleopCfg:
        """Return an Isaac Teleop configuration that uses the embodiment's XR settings and target frame.

        Args:
            pipeline_builder: Isaac Teleop pipeline builder. Required.
            embodiment: Embodiment that provides the XR configuration and the teleoperation target frame prim path.
                None selects a standard XR configuration.
        """
        if pipeline_builder is None:
            raise ValueError("OpenXRCfg requires a pipeline_builder (got None)")
        xr_cfg = embodiment.get_xr_cfg() if embodiment is not None else XrCfg()
        target_frame_prim_path = embodiment.get_teleop_target_frame_prim_path()
        return IsaacTeleopCfg(
            pipeline_builder=pipeline_builder,
            sim_device=self.sim_device,
            xr_cfg=xr_cfg,
            target_frame_prim_path=target_frame_prim_path,
        )


@register_device
class KeyboardCfg(TeleopDeviceBase):
    """Keyboard device that commands an SE(3) end-effector motion."""

    name = "keyboard"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        """Create a keyboard device.

        Args:
            sim_device: Device the simulation runs on. None leaves it unspecified.
            pos_sensitivity: Scale of translation commands.
            rot_sensitivity: Scale of rotation commands.
        """
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> Se3KeyboardCfg:
        """Return the Isaac Lab SE(3) keyboard configuration with this device's sensitivities.

        Args:
            pipeline_builder: Unused; the keyboard has no Isaac Teleop pipeline.
            embodiment: Unused.
        """
        return Se3KeyboardCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )


@register_device
class SpaceMouseCfg(TeleopDeviceBase):
    """SpaceMouse device that commands an SE(3) end-effector motion."""

    name = "spacemouse"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        """Create a SpaceMouse device.

        Args:
            sim_device: Device the simulation runs on. None leaves it unspecified.
            pos_sensitivity: Scale of translation commands.
            rot_sensitivity: Scale of rotation commands.
        """
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> Se3SpaceMouseCfg:
        """Return the Isaac Lab SE(3) SpaceMouse configuration with this device's sensitivities.

        Args:
            pipeline_builder: Unused; the SpaceMouse has no Isaac Teleop pipeline.
            embodiment: Unused.
        """
        return Se3SpaceMouseCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )
