# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Plain (no-stand) Franka panda so the robot is ground-mounted at the origin, matching the
# LIBERO floor scene (the default franka_ik USD is mounted on a table-height stand).
_PLAIN_PANDA = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
# LIBERO Franka home configuration: q1..q7 + the two finger joints (open).
_HOME_Q = [0.0, -0.16104, 0.0, -2.4446, 0.0, 2.22675, 0.7854, 0.04, 0.04]
# Comfortable reach box in front of the (ground-mounted) base for the groceries.
_REACH_BOX = dict(x_min=0.12, x_max=0.52, y_min=-0.30, y_max=0.26)


@register_environment
class LiberoObjectPackingEnvironment(ExampleEnvironmentBase):
    """LIBERO grocery-packing scene on the floor, placed by the relation solver.

    A basket plus six HOPE groceries; positions are solved (On a thin invisible surface,
    bounded to the Franka reach box, jittered per reset) rather than hardcoded.
    """

    name = "libero_object_packing"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits, RandomAroundSolution
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
        from isaaclab_arena.utils.pose import Pose

        def q(w, x, y, z):  # MuJoCo wxyz -> Arena xyzw
            return (x, y, z, w)

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(1500.0)
        ground = self.asset_registry.get_asset_by_name("ground_plane")()

        # Ground-mounted Franka at the origin with the LIBERO home pose.
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.20, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        embodiment.set_initial_joint_pose(initial_joint_pose=_HOME_Q)
        embodiment.scene_config.robot.spawn.usd_path = _PLAIN_PANDA

        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)() if args_cli.teleop_device else None
        )

        # Invisible thin surface at floor level: the solver anchor + On() parent (gives objects Z).
        surface = self.asset_registry.get_asset_by_name("procedural_table")()
        surface.set_initial_pose(Pose(position_xyz=(0.32, 0.0, -0.02), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        surface.bounding_box = AxisAlignedBoundingBox(min_point=(-0.4, -0.75, -0.02), max_point=(0.4, 0.75, 0.02))
        surface.add_relation(IsAnchor())

        # Basket: static asset (kept out of the 2D solve so it keeps its opening-up orientation).
        basket = self.asset_registry.get_asset_by_name(args_cli.basket)()
        basket.set_initial_pose(Pose(position_xyz=(0.28, 0.42, 0.0), rotation_xyzw=q(0.7071, -0.0017, 0.0017, 0.7071)))

        # Groceries: relation-solved placement (On surface, within reach, jittered per reset).
        objects = []
        for obj_name in args_cli.objects:
            obj = self.asset_registry.get_asset_by_name(obj_name)()
            obj.add_relation(On(surface, edge_margin_m=0.03))
            obj.add_relation(PositionLimits(**_REACH_BOX))
            obj.add_relation(RandomAroundSolution(x_half_m=0.04, y_half_m=0.04, yaw_half_rad=0.4))
            objects.append(obj)

        scene = Scene(assets=[ground, light, surface, basket, *objects])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--objects",
            nargs="*",
            default=[
                "alphabet_soup_can_hope_robolab",
                "tomato_sauce_can_hope_robolab",
                "milk_carton_hope_robolab",
                "salad_dressing_bottle_hot3d_robolab",
                "cream_cheese_hope_robolab",
                "butter_hope_robolab",
            ],
            help="grocery assets to pack (relation-solved placement)",
        )
        parser.add_argument("--basket", type=str, default="grey_bin_robolab", help="container asset")
        parser.add_argument("--teleop_device", type=str, default=None)
