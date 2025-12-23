# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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


class CubeGoalPoseEnvironment(ExampleEnvironmentBase):
    """
    A environment for achieving the goal pose of a cube.
    """

    name = "cube_goal_pose"

    def get_env(self, args_cli: argparse.Namespace):

        from isaaclab.envs.mdp.events import reset_scene_to_default
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.utils import configclass
        from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.utils.pose import Pose

        @configclass
        class EventCfg:
            """Configuration for events."""

            reset_all = EventTerm(func=reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

            init_franka_arm_pose = EventTerm(
                func=franka_stack_events.set_default_joint_pose,
                mode="reset",
                params={
                    "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
                },
            )

        # Add the asset registry from the arena migration package
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        light = self.asset_registry.get_asset_by_name("light")()
        object = self.asset_registry.get_asset_by_name(args_cli.object)()
        object.set_initial_pose(
            Pose(
                position_xyz=(0.1, 0.0, 0.2),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.4, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
            # increase sensitivity for teleop device
            teleop_device.pos_sensitivity = 0.25
            teleop_device.rot_sensitivity = 0.5
        else:
            teleop_device = None

        object_thresholds = {
            "success_zone": {
                "z_range": [0.2, 1],
            },
            "orientation": {
                "target": [0.7071, 0.0, 0.0, 0.7071],  # yaw 90 degrees
                "tolerance_rad": 0.2,
            },
        }

        scene = Scene(assets=[background, light, object])

        task = GoalPoseTask(object, object_thresholds=object_thresholds)

        # add custom randomization events of the initial objectposes
        task.events_cfg = EventCfg()

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
        parser.add_argument("--object", type=str, default="dex_cube")
        parser.add_argument("--background", type=str, default="table")
        parser.add_argument("--embodiment", type=str, default="franka")
        parser.add_argument("--enable_cameras", type=bool, default=False)
        parser.add_argument("--teleop_device", type=str, default=None)
