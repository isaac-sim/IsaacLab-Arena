# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
class Gr1TurnStandMixerKnobEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1 stand-mixer-knob environment."""

    enable_cameras: bool = False
    object: str | None = None
    teleop_device: str | None = None
    embodiment: str = "gr1_pink"
    target_level: int = 4
    reset_level: int = -1


@register_environment
class Gr1TurnStandMixerKnobEnvironment(ExampleEnvironmentBase[Gr1TurnStandMixerKnobEnvironmentCfg]):

    name: str = "gr1_turn_stand_mixer_knob"
    _legacy_argparse_cfg_type = Gr1TurnStandMixerKnobEnvironmentCfg

    def build(self, cfg: Gr1TurnStandMixerKnobEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.turn_knob_task import TurnKnobTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        stand_mixer = self.asset_registry.get_asset_by_name("stand_mixer")()
        assets = [background, stand_mixer]
        assert cfg.embodiment in ["gr1_pink", "gr1_joint"], f"Invalid GR1T2 embodiment {cfg.embodiment}"
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Put the microwave on the packing table.
        stand_mixer_pose = Pose(
            position_xyz=(0.4, -0.00586, 0.22773),
            rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
        )
        stand_mixer.set_initial_pose(stand_mixer_pose)

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
            task=TurnKnobTask(turnable_object=stand_mixer, target_level=cfg.target_level, reset_level=cfg.reset_level),
            teleop_device=teleop_device,
        )

        return isaaclab_arena_environment
