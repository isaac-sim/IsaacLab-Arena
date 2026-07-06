# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class PickAndPlaceMapleTableEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Maple-table pick-and-place environment."""

    enable_cameras: bool = False
    embodiment: str = "droid_abs_joint_pos"
    hdr: str | None = None
    light_intensity: float = 500.0
    pick_up_object: str = "rubiks_cube_hot3d_robolab"
    destination_location: str = "bowl_ycb_robolab"
    additional_table_objects: list[str] = field(default_factory=list)


@register_environment
class PickAndPlaceMapleTableEnvironment(ArenaEnvironmentFactory[PickAndPlaceMapleTableEnvironmentCfg]):
    """Registered provider for the Maple-table pick-and-place environment."""

    name: str = "pick_and_place_maple_table"
    _legacy_argparse_cfg_type = PickAndPlaceMapleTableEnvironmentCfg

    def build(self, cfg: PickAndPlaceMapleTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.pick_up_object)()
        destination_location = self.asset_registry.get_asset_by_name(cfg.destination_location)()

        # Step 2: Describe spatial relationships
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        pick_up_object.add_relation(On(table_reference))
        destination_location.add_relation(On(table_reference))

        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in cfg.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Step 3: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=cfg.light_intensity),
        )
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())

        # Step 4: Select the embodiment
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )

        # Step 5: Compose the scene
        scene = Scene(
            assets=[background, light, pick_up_object, destination_location, table_reference, *additional_table_objects]
        )

        # Step 6: Define the task
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            episode_length_s=20.0,
        )

        # Set viewport camera to match the robolab droid view
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        # Step 7: Assemble the environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
        )
        return isaaclab_arena_environment

    # TODO(cvolk, 2026-07-03): Delete this CLI-only option when teleoperation runners
    # receive typed configuration instead of the environment subparser namespace.
    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        # Consumed directly by teleop.py and record_demos.py, not by build(cfg).
        parser.add_argument("--teleop_device", type=str, default=None)
