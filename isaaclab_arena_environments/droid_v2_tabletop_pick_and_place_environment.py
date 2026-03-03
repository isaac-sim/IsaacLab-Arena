# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID v2 tabletop pick-and-place environment using the droid_mimic_fixed embodiment.

Uses SortMultiObjectTask so that success termination only fires when ALL
pickable objects are placed in the bin, preventing premature env resets
during multi-object scripted pick-and-place.
"""

import argparse
import math
import random

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Randomize yaw radian
# random_yaw_radian = math.radians(random.randint(0, 360))
random_yaw_radian = math.radians(-120)

class DroidV2TabletopPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """DROID v2 environment with flattened USD and mimic joint constraints for the Robotiq 2F-85 gripper."""

    name: str = 'droid_v2_tabletop_pick_and_place'

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        """Build and return the IsaacLab Arena environment."""
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
        import isaaclab.sim as sim_utils
        from isaaclab_arena.relations.relations import (
            AtPosition,
            IsAnchor,
            NextTo,
            On,
            RandomAroundSolution,
            RotateAroundSolution,
            Side,
        )
        from isaaclab_arena.utils.pose import Pose, PoseRange

        office_table = self.asset_registry.get_asset_by_name('office_table_background')()
        ground_plane = self.asset_registry.get_asset_by_name('ground_plane')()
        obj_1 = self.asset_registry.get_asset_by_name('tomato_soup_can')(scale=(0.7, 0.7, 0.6))
        obj_2 = self.asset_registry.get_asset_by_name('ketchup_bottle_hope_robolab')(scale=(0.6, 0.6, 0.5))
        obj_3 = self.asset_registry.get_asset_by_name('alphabet_soup_can_hope_robolab')(scale=(0.7, 0.7, 0.8))

        blue_sorting_bin = self.asset_registry.get_asset_by_name('blue_sorting_bin')(scale=(2.0, 0.8, 2.0))
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name('light')(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name('droid_differential_ik')(enable_cameras=args_cli.enable_cameras)

        office_table.set_initial_pose(Pose(position_xyz=(0.7, 0.5, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0)))
        embodiment.set_initial_pose(Pose(position_xyz=(0.1, 0.18, 0.75), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        blue_sorting_bin.set_initial_pose(
            Pose(
                position_xyz=(0.67, 0.4, 0.8),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        office_table.add_relation(IsAnchor())
        blue_sorting_bin.add_relation(IsAnchor())
        obj_1.add_relation(On(office_table))
        obj1_distance_to_blue_sorting_bin = random.uniform(0.2, 0.3)
        obj_1.add_relation(NextTo(blue_sorting_bin, side=Side.NEGATIVE_Y, distance_m=obj1_distance_to_blue_sorting_bin))
        # obj_1.add_relation(
        #     AtPosition(x=0.5, y=0.1, z=0.86)
        # )
        obj_1.add_relation(RotateAroundSolution(roll_rad=1.5707963, yaw_rad=0))
        obj_2.add_relation(On(office_table))
        obj2_distance_to_obj1 = random.uniform(0.05, 0.15)
        obj_2.add_relation(NextTo(obj_1, side=Side.NEGATIVE_X, distance_m=obj2_distance_to_obj1))
        obj_2.add_relation(RotateAroundSolution(yaw_rad=random_yaw_radian))
        obj_3.add_relation(On(office_table))
        obj3_distance_to_blue_sorting_bin = random.uniform(0.05, 0.15)
        obj_3.add_relation(NextTo(blue_sorting_bin, side=Side.POSITIVE_Y, distance_m=obj3_distance_to_blue_sorting_bin))
        # obj_3.add_relation(RotateAroundSolution(roll_rad=1.5707963))


        # obj_1.set_initial_pose(
        #     PoseRange(
        #         position_xyz_min=(0.5, 0.1, 0.86),
        #         position_xyz_max=(0.6, -0.12, 0.86),
        #         rpy_min=(-1.5707963, 0.0, -1.5707963),
        #         rpy_max=(-1.5707963, 0.0, -1.5707963),
        #     )
        # )
        # obj_2.set_initial_pose(
        #     PoseRange(
        #         position_xyz_min=(0.4, -0.1, 0.86),
        #         position_xyz_max=(0.6, -0.2, 0.86),
        #         rpy_min=(0, 0, math.radians(-111.55)),
        #         rpy_max=(0, 0, math.radians(-111.55)),
        #     )
        # )
        # obj_3.set_initial_pose(
        #     PoseRange(
        #         position_xyz_min=(0.5, 0.7, 0.86),
        #         position_xyz_max=(0.7, 0.8, 0.86),
        #         rpy_min=(1.5707963, 0, 0),
        #         rpy_max=(1.5707963, 0, 0),
        #     )
        # )

        # Shared destination for all objects
        destination_location = ObjectReference(
            name='destination_location',
            prim_path='{ENV_REGEX_NS}/blue_sorting_bin/Geometry/sm_bin_20x25x05cm_a01_01',
            parent_asset=blue_sorting_bin,
            object_type=ObjectType.RIGID,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        assets = [office_table, ground_plane, obj_1, obj_2, obj_3, blue_sorting_bin, light]


        scene = Scene(assets=assets)

        # All pickable objects share the same destination (the bin).
        # SortMultiObjectTask creates a contact sensor per object and only
        # fires the success termination when ALL objects are on the destination.
        pick_up_objects = [obj_1, obj_2, obj_3]
        destinations = [destination_location] * len(pick_up_objects)

        task = SortMultiObjectTask(
            pick_up_object_list=pick_up_objects,
            destination_location_list=destinations,
            background_scene=office_table,
            episode_length_s=600.0,
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
        """Add CLI arguments specific to this environment."""
        parser.add_argument('--object', type=str, default='tomato_soup_can')
        parser.add_argument('--object_set', nargs='+', type=str, default=None)
        parser.add_argument('--teleop_device', type=str, default=None)
