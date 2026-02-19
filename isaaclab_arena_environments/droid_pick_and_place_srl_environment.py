# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class DroidPickAndPlaceSRLEnvironment(ExampleEnvironmentBase):

    name: str = "droid_pick_and_place_srl"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_library import ISAACLAB_STAGING_NUCLEUS_DIR
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        background = self.asset_registry.get_asset_by_name("rubiks_cube_bowl_srl")()
        rubiks_cube = ObjectReference(
            name="rubiks_cube",
            prim_path="{ENV_REGEX_NS}/rubiks_cube_bowl_srl/rubiks_cube",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        bowl = ObjectReference(
            name="bowl",
            prim_path="{ENV_REGEX_NS}/rubiks_cube_bowl_srl/bowl",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )

        light_spawner_cfg = sim_utils.DomeLightCfg(
            texture_file=f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/home_office.exr",
            intensity=500.0,
            visible_in_primary_ray=True,
            texture_format="latlong",
        )
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )

        teleop_device = None
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        scene = Scene(assets=[background, light, rubiks_cube, bowl])
        task = PickAndPlaceTask(
            pick_up_object=rubiks_cube,
            destination_location=bowl,
            background_scene=background,
        )

        # Set viewport camera to match the robolab droid view
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_set_viewer_cfg,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="droid_abs_joint_pos")
        parser.add_argument("--teleop_device", type=str, default=None)
