# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class PressButtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the press-button environment."""

    object: str | None = None
    teleop_device: str | None = None
    embodiment: str = "franka_ik"


@register_environment
class PressButtonEnvironment(ExampleEnvironmentBase[PressButtonEnvironmentCfg]):

    name: str = "press_button"
    _legacy_argparse_cfg_type = PressButtonEnvironmentCfg

    def build(self, cfg: PressButtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.press_button_task import PressButtonTask
        from isaaclab_arena.utils.pose import Pose

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)()

        background = self.asset_registry.get_asset_by_name("packing_table")()
        press_object = self.asset_registry.get_asset_by_name("coffee_machine")()

        assets = [background, press_object]

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Put the coffee_machine on the packing table.
        press_object_pose = Pose(position_xyz=(0.7, 0.4, 0.19), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
        press_object.set_initial_pose(press_object_pose)

        # Compose the scene
        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PressButtonTask(press_object, reset_pressedness=0.8),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment
