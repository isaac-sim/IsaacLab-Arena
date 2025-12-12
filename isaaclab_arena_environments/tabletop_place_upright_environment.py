# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class TableTopPlaceUprightEnvironment(ExampleEnvironmentBase):
    """
    A place upright environment for the Seattle Lab table.
    """

    name = "tabletop_place_upright"
    
    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.object_library import GroundPlane, Light
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import configclass
        import isaaclab.envs.mdp as mdp
        from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import randomize_object_pose
        from isaaclab_arena.embodiments.agibot.agibot import AgibotEmbodiment
        from isaaclab_arena.tasks.place_upright_task import PlaceUprightTask

        @configclass
        class EventCfgPlaceUprightMug:
            """Configuration for events."""
            reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

            randomize_mug_positions = EventTerm(
                func=randomize_object_pose,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.05, 0.2),
                        "y": (-0.10, 0.10),
                        "z": (0.75, 0.75),
                        "roll": (-1.57, -1.57),
                        "yaw": (-0.57, 0.57),
                    },
                    "asset_cfgs": [SceneEntityCfg("mug")],
                },
            )

        # Add the asset registry from the arena migration package
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        placeable_object = self.asset_registry.get_asset_by_name(args_cli.object)(initial_pose=Pose(position_xyz=(0.05, 0.0, 0.75), rotation_wxyz=(0.0, 1.0, 0.0, 0.0)))
        if args_cli.embodiment in ["agibot", "galbot"]:
            embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras, arm_mode="left")
        else:
            raise NotImplementedError

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None
        
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.60, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        scene = Scene(assets=[background, placeable_object, ground_plane, light])
        
        task = PlaceUprightTask(placeable_object, placeable_object.orientation_threshold)
        task.events_cfg = EventCfgPlaceUprightMug()

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
        parser.add_argument("--object", type=str, default="mug")
        parser.add_argument("--background", type=str, default="place_upright_mug_table") 
        parser.add_argument("--embodiment", type=str, default="agibot")
        parser.add_argument("--enable_cameras", type=bool, default=False)
        parser.add_argument("--teleop_device", type=str, default=None)

