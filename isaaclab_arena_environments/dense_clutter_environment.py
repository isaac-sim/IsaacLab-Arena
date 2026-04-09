# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Dense clutter environment demonstrating mesh vs. AABB collision optimisation.

Four tomato-soup cans are arranged in a tight 2×2 offset grid on a table.
Each diagonal pair is separated by 5 cm in both X and Y.  Because the cans
are cylindrical, their axis-aligned bounding boxes (squares in the XY
plane) overlap at the corners, but the actual round meshes do **not**
touch: the Euclidean distance between centres (~7.1 cm) exceeds the can
diameter (~6.6 cm).

**Mesh collision mode** — SDF queries confirm no mesh penetration; the
solver keeps the cans in the original tight packing:

  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \\
    --viz kit --policy_type zero_action --num_steps 500 \\
    --collision_mode mesh \\
    dense_clutter --embodiment gr1_joint

**AABB collision mode** — the overlap-volume loss detects the bounding-box
overlap and pushes diagonal neighbours apart, yielding a more spread-out
arrangement:

  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \\
    --viz kit --policy_type zero_action --num_steps 500 \\
    --collision_mode aabb \\
    dense_clutter --embodiment gr1_joint

A cracker box and a mustard bottle are placed on opposite sides of the
table for visual context; they are far enough apart that neither mode
affects them.
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Tight offset grid of cylindrical cans.
# Diagonal offset of 0.05 m in X and Y gives:
#   AABB overlap per axis ≈ 0.066 − 0.05 = 0.016 m  (overlapping)
#   Mesh distance = 0.05·√2 ≈ 0.071 m > 0.066 m      (not touching)
CAN_POSITIONS = [
    ("can_a", (-0.075, -0.025)),
    ("can_b", (0.025, -0.025)),
    ("can_c", (-0.025, 0.025)),
    ("can_d", (0.075, 0.025)),
]


class DenseClutterEnvironment(ExampleEnvironmentBase):
    """Table scene with cylindrical cans in tight diagonal packing.

    Diagonal neighbours have overlapping AABBs (square bounding boxes)
    but non-overlapping meshes (round cross-sections).  Mesh collision
    mode preserves the tight arrangement; AABB collision mode spreads the
    cans out to eliminate bounding-box overlap.
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

        # Tightly packed cans at the table centre.
        for instance_name, (cx, cy) in CAN_POSITIONS:
            can = self.asset_registry.get_asset_by_name("tomato_soup_can")(instance_name=instance_name)
            can.add_relation(On(tabletop_reference))
            can.set_initial_pose(Pose(position_xyz=(cx, cy, 0.0)))
            placeable_assets.append(can)

        # Context objects placed far from the cans (no AABB overlap with anything).
        for obj_name, (ox, oy) in [("cracker_box", (-0.40, 0.0)), ("mustard_bottle", (0.40, 0.0))]:
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
