# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class G1AgileTabletopAppleToPlateEnvironment(ExampleEnvironmentBase):
    """G1 robot with WBC-AGILE policy doing tabletop apple-to-plate manipulation.

    The robot stands stationary at a table and moves an apple onto a plate.
    The AGILE whole-body-control policy handles balance while the upper body
    performs manipulation via direct joint control.
    """

    name: str = "g1_agile_tabletop_apple_to_plate"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)(scale=(0.01, 0.01, 0.01))
        destination_location = self.asset_registry.get_asset_by_name("clay_plates_hot3d_robolab")(scale=(0.5, 0.5, 0.5))
        import isaaclab.sim as sim_utils

        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0)
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Create a reference to the table surface for spatial relations.
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        from isaaclab_arena.utils.pose import Pose

        # Place objects on the table surface; On() handles z-height automatically.
        pick_up_object.set_initial_pose(Pose(position_xyz=(0.8, 0.1, 0.0)))
        pick_up_object.add_relation(On(table_reference))
        destination_location.set_initial_pose(Pose(position_xyz=(0.8, -0.1, 0.0)))
        destination_location.add_relation(On(table_reference))

        embodiment.set_initial_pose(
            Pose(
                position_xyz=(1.2, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
            )
        )

        scene = Scene(assets=[background, pick_up_object, destination_location, table_reference, light])
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            episode_length_s=30.0,
            task_description="Pick up the apple from the table and place it onto the plate.",
        )
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="apple_01_objaverse_robolab")
        parser.add_argument("--embodiment", type=str, default="g1_wbc_agile_joint")
        parser.add_argument("--teleop_device", type=str, default=None)
