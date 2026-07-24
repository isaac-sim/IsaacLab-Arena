# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class DeformableSpherePickPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the deformable Maple-table pick-and-place smoke environment."""

    enable_cameras: bool = False
    embodiment: str = "droid_abs_joint_pos"
    light_intensity: float = 500.0
    pick_up_object: str = "procedural_deformable_sphere"
    destination_location: str = "bowl_ycb_robolab"


@register_environment
class DeformableSpherePickPlaceEnvironment(ArenaEnvironmentFactory[DeformableSpherePickPlaceEnvironmentCfg]):
    """Registered provider for a deformable pick-and-place scene on the Maple table."""

    name: str = "deformable_sphere_pick_place"
    _legacy_argparse_cfg_type = DeformableSpherePickPlaceEnvironmentCfg

    def build(self, cfg: DeformableSpherePickPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.deformable_pick_and_place_task import DeformablePickAndPlaceTask

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.pick_up_object)()
        destination_location = self.asset_registry.get_asset_by_name(cfg.destination_location)()

        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())
        pick_up_object.add_relation(On(table_reference))
        destination_location.add_relation(On(table_reference))

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)

        scene = Scene(
            assets=[
                background,
                light,
                directional_light,
                pick_up_object,
                destination_location,
                table_reference,
            ]
        )
        task = DeformablePickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            episode_length_s=20.0,
            max_separation=(0.12, 0.12, 0.16),
        )

        def _set_viewer_cfg(env_cfg):
            # replicate_physics is set centrally by ArenaEnvBuilder based on the selected physics
            # backend (True for Newton, False for PhysX deformables), so it is not touched here.
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
            placer_params=ObjectPlacerParams(resolve_on_reset=False),
        )
