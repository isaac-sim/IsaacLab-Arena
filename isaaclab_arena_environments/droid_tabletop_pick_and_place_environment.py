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


class DroidTabletopPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "droid_tabletop_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        import isaaclab.sim as sim_utils
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("table")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        blue_sorting_bin = self.asset_registry.get_asset_by_name("blue_sorting_bin")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name("droid")(enable_cameras=args_cli.enable_cameras)

        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.45, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        blue_sorting_bin.set_initial_pose(
            Pose(
                position_xyz=(0.25, 0.1, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # TODO(alexmillane, 2025.09.24): Add automatic object type detection of ObjectReferences.
        # destination_location = ObjectReference(
        #     name="destination_location",
        #     parent_asset=blue_sorting_bin,
        #     object_type=ObjectType.RIGID,
        # )
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        if args_cli.object_set is not None and len(args_cli.object_set) > 0:
            objects = []
            for obj in args_cli.object_set:
                obj_from_set = self.asset_registry.get_asset_by_name(obj)()
                objects.append(obj_from_set)
            object_set = RigidObjectSet(name="object_set", objects=objects)
            object_set.set_initial_pose(Pose(position_xyz=(0.4, 0.2, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
            scene = Scene(assets=[background, pick_up_object,  blue_sorting_bin, object_set, light])

        else:
            scene = Scene(assets=[background, pick_up_object, blue_sorting_bin, light])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            # task=PickAndPlaceTask(pick_up_object, destination_location, background),
            task = NoTask(),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="tomato_soup_can")
        parser.add_argument("--object_set", nargs="+", type=str, default=None)
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
