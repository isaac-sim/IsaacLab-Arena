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
class GalileoPickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Galileo pick-and-place environment."""

    enable_cameras: bool = False
    object: str = "power_drill"
    embodiment: str = "gr1_pink"
    teleop_device: str | None = None


@register_environment
class GalileoPickAndPlaceEnvironment(ExampleEnvironmentBase[GalileoPickAndPlaceEnvironmentCfg]):

    name: str = "galileo_pick_and_place"
    _legacy_argparse_cfg_type = GalileoPickAndPlaceEnvironmentCfg

    def build(self, cfg: GalileoPickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("galileo")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.55, 0.0, 0.33),
                rotation_xyzw=(0.0, 0.7071068, -0.7071068, 0.0),
            )
        )

        # NOTE(alexmillane, 2025.09.08): This is a sub-optimal destination location
        # in the room. I'd like to use the bottom shelf, however, the whole shelf is
        # a single prim and therefore I cannot pick out the bottom shelf specifically.
        # NOTE(alexmillane, 2025.09.08): I've also had to apply the rigid body API to
        # the lid via the UI.
        # TODO(alexmillane, 2025.09.08): Separate the self into prims so we can reference
        # the bottom shelf specifically.
        destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/galileo/BackgroundAssets/bins/small_bin_grid_01/lid",
            parent_asset=background,
        )

        scene = Scene(assets=[background, pick_up_object, destination_location])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(pick_up_object, destination_location, background),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment
