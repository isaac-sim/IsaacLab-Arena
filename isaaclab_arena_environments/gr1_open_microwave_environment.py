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
class Gr1OpenMicrowaveEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1 open-microwave environment."""

    enable_cameras: bool = False
    object: str | None = None
    teleop_device: str | None = None
    embodiment: str = "gr1_pink"


@register_environment
class Gr1OpenMicrowaveEnvironment(ExampleEnvironmentBase[Gr1OpenMicrowaveEnvironmentCfg]):

    name: str = "gr1_open_microwave"
    _legacy_argparse_cfg_type = Gr1OpenMicrowaveEnvironmentCfg

    def build(self, cfg: Gr1OpenMicrowaveEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        assets = [background, microwave]
        assert cfg.embodiment in ["gr1_pink", "gr1_joint"], f"Invalid GR1T2 embodiment {cfg.embodiment}"
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Put the microwave on the packing table.
        microwave_pose = Pose(
            position_xyz=(0.4, -0.00586, 0.22773),
            rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
        )
        microwave.set_initial_pose(microwave_pose)

        # Optionally add another object
        if cfg.object is not None:
            object = self.asset_registry.get_asset_by_name(cfg.object)()
            object_pose = Pose(
                position_xyz=(0.466, -0.437, 0.154),
                rotation_xyzw=(-0.5, 0.5, -0.5, 0.5),
            )
            object.set_initial_pose(object_pose)
            assets.append(object)

        # Compose the scene
        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2, episode_length_s=5.0),
            teleop_device=teleop_device,
        )

        return isaaclab_arena_environment
