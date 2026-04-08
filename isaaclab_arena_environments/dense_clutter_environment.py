# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Dense clutter environment demonstrating mesh-level collision checking.

A red container sits at the table centre.  A tomato-soup can and a mug are
placed **inside** the container using explicit ``initial_pose`` values.
Because the container is hollow (open box), their triangle meshes do not
collide — but their axis-aligned bounding boxes overlap the container's
solid AABB.

Run with ``--solver_max_iters 0`` to skip optimisation and evaluate the
hand-placed arrangement directly:

Mesh mode (objects inside the container are accepted):
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \\
    --viz kit --policy_type zero_action --num_steps 500 \\
    --collision_mode mesh --solver_max_iters 0 \\
    dense_clutter --embodiment gr1_joint

AABB mode (same arrangement is rejected — AABBs overlap):
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \\
    --viz kit --policy_type zero_action --num_steps 500 \\
    --collision_mode aabb --solver_max_iters 0 \\
    dense_clutter --embodiment gr1_joint
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Container at half scale in XY, full scale in Z → 9 cm wall height.
# Inner dimensions ~29.6 × 19.6 cm — spacious enough for small objects.
CONTAINER_SCALE = (0.5, 0.5, 1.0)

# Objects placed inside the container.  clearance_m=0.005 lifts them 5 mm
# above the table surface so they clear the container's thin bottom plate.
# XY positions are well inside the container half-extents (0.153, 0.103).
OBJECTS_INSIDE_CONTAINER = [
    ("tomato_soup_can", "can_in_container", (0.05, 0.0)),
    ("tomato_soup_can", "can2_in_container", (-0.05, 0.0)),
]
INSIDE_CLEARANCE_M = 0.03

# Objects on the table with explicit positions (no random placement).
# Spread far enough apart that no AABBs overlap.
OBJECTS_ON_TABLE = [
    ("beer_bottle", (0.45, 0.0)),
    ("mustard_bottle", (-0.45, 0.0)),
    ("cracker_box", (0.0, 0.25)),
    ("sugar_box", (0.0, -0.25)),
]


class DenseClutterEnvironment(ExampleEnvironmentBase):
    """Table scene with objects placed inside a hollow container.

    The container's AABB is a solid box, but its mesh is an open basket.
    Objects inside have AABB overlap (solid-box assumption) but no mesh
    collision (they sit in the hollow interior).

    With ``--collision_mode mesh --solver_max_iters 0`` the arrangement
    passes validation.  With ``--collision_mode aabb --solver_max_iters 0``
    the same arrangement is rejected because the AABBs overlap.
    """

    name: str = "dense_clutter"

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

        tabletop_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
            parent_asset=table_background,
        )
        tabletop_reference.add_relation(IsAnchor())

        placeable_assets = []

        # Container at table centre with explicit initial_pose.
        container = self.asset_registry.get_asset_by_name("red_container")(scale=CONTAINER_SCALE)
        container.add_relation(On(tabletop_reference))
        container.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
        placeable_assets.append(container)

        # Objects inside the container — lifted 5 mm above the table to clear
        # the container's bottom plate mesh.
        for asset_name, instance_name, (ix, iy) in OBJECTS_INSIDE_CONTAINER:
            obj = self.asset_registry.get_asset_by_name(asset_name)(instance_name=instance_name)
            obj.add_relation(On(tabletop_reference, clearance_m=INSIDE_CLEARANCE_M))
            obj.set_initial_pose(Pose(position_xyz=(ix, iy, 0.0)))
            placeable_assets.append(obj)

        # Objects spread around the table at explicit positions (no randomness).
        for obj_name, (ox, oy) in OBJECTS_ON_TABLE:
            obj = self.asset_registry.get_asset_by_name(obj_name)()
            obj.add_relation(On(tabletop_reference))
            obj.set_initial_pose(Pose(position_xyz=(ox, oy, 0.0)))
            placeable_assets.append(obj)

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
        parser.add_argument("--embodiment", type=str, default="gr1_joint", help="Robot embodiment to use")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
