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

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


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

        from isaaclab_arena.utils.pose import Pose, PoseRange

        office_table = self.asset_registry.get_asset_by_name('office_table_background')()
        ground_plane = self.asset_registry.get_asset_by_name('ground_plane')()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        ranch_dressing_bottle = self.asset_registry.get_asset_by_name('ranch_dressing_bottle')()
        sugar_box = self.asset_registry.get_asset_by_name('sugar_box')()

        for obj in [pick_up_object, sugar_box]:
            obj.object_cfg.spawn.scale = (0.8, 0.8, 0.8)

        blue_sorting_bin = self.asset_registry.get_asset_by_name('blue_sorting_bin')()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name('light')(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name('droid_mimic_fixed')(enable_cameras=args_cli.enable_cameras)

        office_table.set_initial_pose(Pose(position_xyz=(0.7, 0.5, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0)))
        embodiment.set_initial_pose(Pose(position_xyz=(0.1, 0.18, 0.75), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        pick_up_object.set_initial_pose(
            PoseRange(
                position_xyz_min=(0.5, 0.1, 0.86),
                position_xyz_max=(0.6, -0.12, 0.86),
                rpy_min=(-1.5707963, 0.0, -1.5707963),
                rpy_max=(-1.5707963, 0.0, -1.5707963),
            )
        )
        ranch_dressing_bottle.set_initial_pose(
            PoseRange(
                position_xyz_min=(0.4, -0.1, 0.86),
                position_xyz_max=(0.6, -0.2, 0.86),
                rpy_min=(0, 0, math.radians(-111.55)),
                rpy_max=(0, 0, math.radians(-111.55)),
            )
        )
        sugar_box.set_initial_pose(
            PoseRange(
                position_xyz_min=(0.5, 0.7, 0.86),
                position_xyz_max=(0.7, 0.8, 0.86),
                rpy_min=(1.5707963, 0, 0),
                rpy_max=(1.5707963, 0, 0),
            )
        )
        blue_sorting_bin.set_initial_pose(
            Pose(
                position_xyz=(0.67, 0.4, 0.8),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

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

        assets = [office_table, ground_plane, pick_up_object, sugar_box, ranch_dressing_bottle, blue_sorting_bin, light]

        if args_cli.object_set is not None and len(args_cli.object_set) > 0:
            objects = []
            for obj in args_cli.object_set:
                obj_from_set = self.asset_registry.get_asset_by_name(obj)()
                objects.append(obj_from_set)
            object_set = RigidObjectSet(name='object_set', objects=objects)
            object_set.set_initial_pose(Pose(position_xyz=(0.4, 0.2, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
            assets.append(object_set)

        scene = Scene(assets=assets)

        # All pickable objects share the same destination (the bin).
        # SortMultiObjectTask creates a contact sensor per object and only
        # fires the success termination when ALL objects are on the destination.
        pick_up_objects = [pick_up_object, ranch_dressing_bottle, sugar_box]
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
