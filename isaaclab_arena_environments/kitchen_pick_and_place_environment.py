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


class KitchenPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "kitchen_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        background = self.asset_registry.get_asset_by_name("kitchen")()
        table_top_reference = ObjectReference(
            name="table_top_reference",
            prim_path="{ENV_REGEX_NS}/kitchen/Kitchen_Counter/TRS_Base/TRS_Static/Counter_Top_A",
            parent_asset=background,
        )
        table_top_reference.add_relation(IsAnchor())

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        # Validate mutually exclusive object arguments
        has_object = args_cli.object is not None
        has_object_set = args_cli.object_set is not None and len(args_cli.object_set) > 0
        assert has_object or has_object_set, "Must specify either --object or --object_set."
        assert not (has_object and has_object_set), (
            "Cannot specify both --object and --object_set. Use --object for a single object, "
            "or --object_set for multiple objects across environments where --num_envs == len(object_set)."
        )

        # Create the pick-up object: Either a single object or a set of objects
        if has_object_set:
            objects = [self.asset_registry.get_asset_by_name(obj)() for obj in args_cli.object_set]
            pick_up_object = RigidObjectSet(name="object_set", objects=objects)
        else:
            pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        pick_up_object.add_relation(On(table_top_reference, clearance_m=0.02))
        pick_up_object.add_relation(AtPosition(x=0.4, y=0.0))

        # TODO(alexmillane, 2025.09.24): Add automatic object type detection of ObjectReferences.
        destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=[background, table_top_reference, pick_up_object, destination_location])
        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
        )
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=pick_and_place_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object", type=str, default=None, help="Single object to pick up. Mutually exclusive with --object_set."
        )
        parser.add_argument(
            "--object_set",
            nargs="+",
            type=str,
            default=None,
            help="Multiple objects to spawn across environments. Mutually exclusive with --object.",
        )
        parser.add_argument("--embodiment", type=str, default="franka")
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
