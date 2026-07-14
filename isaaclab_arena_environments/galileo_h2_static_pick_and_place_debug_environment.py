# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug H2 static apple-to-plate scene.

This is a temporary bridge for H2 bringup: it reuses the static apple scene but
keeps the fixed-root H2 debug embodiment and direct joint-position action space.
It is not the final H2 policy/teleop workflow.
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena_environments.galileo_static_apple_scene import (
    TUNED_DESTINATION_NAME,
    TUNED_PICK_UP_OBJECT_NAME,
    build_static_apple_scene_assets,
    make_static_apple_env_cfg_callback,
    make_static_apple_task_description,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# The Galileo background is offset downward; using the standalone H2 root height
# (z=1.05) puts the feet on the shelf. This lowers H2 to the lab floor while
# keeping the same x/y position for visual reach checks.
DEFAULT_H2_DEBUG_ROBOT_XYZ = (0.25, 0.08, 0.25)


def _yaw_quat_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5))


@register_environment
class GalileoH2StaticPickAndPlaceDebugEnvironment(ExampleEnvironmentBase):
    """Static apple-to-plate scene with the H2 debug joint-position embodiment."""

    name: str = "galileo_h2_static_pick_and_place_debug"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp.galileo_h2_static_pick_and_place_debug.robot_configs import (
            H2_STATIC_DEBUG_OPEN_ARM_JOINT_POS,
        )

        scene_assets = build_static_apple_scene_assets(
            self.asset_registry,
            pick_up_object_name=args_cli.object,
            destination_name=args_cli.destination,
            warning_prefix=self.name,
        )
        background = scene_assets.background
        shelf_support = scene_assets.shelf_support
        pick_up_object = scene_assets.pick_up_object
        destination = scene_assets.destination

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(args_cli.robot_x, args_cli.robot_y, args_cli.robot_z),
                rotation_xyzw=_yaw_quat_xyzw(args_cli.robot_yaw),
            )
        )
        if args_cli.open_arm_pose:
            embodiment.set_joint_initial_pos(H2_STATIC_DEBUG_OPEN_ARM_JOINT_POS)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        task_description = args_cli.task_description
        if task_description is None:
            task_description = make_static_apple_task_description(args_cli.object, args_cli.destination)

        scene = Scene(assets=[background, shelf_support, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                background_scene=background,
                episode_length_s=args_cli.episode_length_s,
                task_description=task_description,
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=make_static_apple_env_cfg_callback(),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default=TUNED_PICK_UP_OBJECT_NAME)
        parser.add_argument("--destination", type=str, default=TUNED_DESTINATION_NAME)
        parser.add_argument("--embodiment", type=str, default="h2_debug_joint_pos")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--task_description", type=str, default=None)
        parser.add_argument("--episode_length_s", type=float, default=6.0)
        parser.add_argument("--robot_x", type=float, default=DEFAULT_H2_DEBUG_ROBOT_XYZ[0])
        parser.add_argument("--robot_y", type=float, default=DEFAULT_H2_DEBUG_ROBOT_XYZ[1])
        parser.add_argument("--robot_z", type=float, default=DEFAULT_H2_DEBUG_ROBOT_XYZ[2])
        parser.add_argument("--robot_yaw", type=float, default=0.0)
        parser.add_argument(
            "--open_arm_pose",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Apply a mild H2 shoulder-open reset pose for the debug shelf scene. "
                "Pass --no-open_arm_pose to use the neutral H2 joint pose."
            ),
        )
