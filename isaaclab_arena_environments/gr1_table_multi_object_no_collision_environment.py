# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Table + multi-object no-overlap environment. Office table with objects placed via
On(table) with built-in no-overlap (relation solver). Includes a robot (e.g. GR1).
No task -- suitable for policy_runner with zero_action or any policy.

Example:
  python isaaclab_arena/evaluation/policy_runner.py --viz kit --policy_type zero_action --num_steps 500 \\
    --num_envs 16 --env_spacing 4.0 --enable_cameras gr1_table_multi_object_no_collision --embodiment gr1_joint
"""

import argparse

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

DEFAULT_TABLE_OBJECTS = [
    "cracker_box",
    "sugar_box",
    "tomato_soup_can",
    "dex_cube",
    "power_drill",
    "red_container",
]
# NOTE: The gradient-based solver does not guarantee collision-free placement for all
# objects. Better initialization strategies and constraining unchanged pose dimensions
# are needed in the near future.


@register_environment
class GR1TableMultiObjectNoCollisionEnvironment(ExampleEnvironmentBase):
    """
    Table-based scene with multiple objects (On(table) + built-in no-overlap) and a robot.
    Layout is solved by ArenaEnvBuilder default relation solving; reset uses asset events.
    """

    name: str = "gr1_table_multi_object_no_collision"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        enable_cameras = getattr(args_cli, "enable_cameras", False)
        camera_offset = Pose(
            position_xyz=(0.12515, 0.0, 0.06776),
            rotation_xyzw=(0.11204, -0.17712, -0.79108, 0.57469),
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=enable_cameras,
            camera_offset=camera_offset,
            use_tiled_camera=(getattr(args_cli, "num_envs", 1) > 1),
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(1.2, 0.0, 0.995),
                rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
            )
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        table_background = self.asset_registry.get_asset_by_name("office_table")()
        light = self.asset_registry.get_asset_by_name("light")()

        # Table surface as anchor for On relations
        tabletop_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
            parent_asset=table_background,
        )
        tabletop_reference.add_relation(IsAnchor())

        object_names = getattr(args_cli, "objects", None) or DEFAULT_TABLE_OBJECTS
        placeable_assets = []
        for name in object_names:
            obj = self.asset_registry.get_asset_by_name(name)()
            obj.add_relation(On(tabletop_reference))
            placeable_assets.append(obj)
        # No-overlap between all pairs is handled automatically by the solver (built-in behavior).

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(
            assets=[
                ground_plane,
                table_background,
                tabletop_reference,
                *placeable_assets,
                light,
            ]
        )
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--objects",
            nargs="*",
            type=str,
            default=None,
            help=f"Object names to spawn on the table (On + no-overlap). Default: {' '.join(DEFAULT_TABLE_OBJECTS)}",
        )
        parser.add_argument("--embodiment", type=str, default="gr1_joint", help="Robot embodiment to use")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
