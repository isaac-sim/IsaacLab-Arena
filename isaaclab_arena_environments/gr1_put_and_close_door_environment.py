# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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


class GR1PutAndCloseDoorEnvironment(ExampleEnvironmentBase):
    """
    A sequential task environment with two subtasks for GR1 humanoid robot:
    1. Pick and place vegetable into the refrigerator shelf
    2. Close the refrigerator door
    The refrigerator starts open, the robot places the object inside, then closes it.
    Uses the lightwheel robocasa kitchen background.
    """

    name = "put_item_in_fridge_and_close_door"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
        from isaaclab.utils import configclass

        from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
        from isaaclab_arena.tasks.task_base import TaskBase
        from isaaclab_arena.utils.pose import Pose

        # Custom task class for this environment
        class PutAndCloseDoorTask(SequentialTaskBase):
            def __init__(
                self,
                subtasks: list[TaskBase],
                episode_length_s: float | None = None,
            ):
                super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

            def get_viewer_cfg(self):
                return self.subtasks[0].get_viewer_cfg()

            def get_prompt(self):
                return None

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                mimic_env_cfg = PutAndCloseDoorTaskMimicEnvCfg()
                mimic_env_cfg.subtask_configs = self.combine_mimic_subtask_configs(ArmMode.RIGHT)

                # Override default subtask term offset range and action noise
                for eef_name, subtask_list in mimic_env_cfg.subtask_configs.items():
                    for subtask_config in subtask_list:
                        subtask_config.subtask_term_offset_range = (0, 0)
                        subtask_config.action_noise = 0.001

                return mimic_env_cfg

        @configclass
        class PutAndCloseDoorTaskMimicEnvCfg(MimicEnvCfg):
            """
            Isaac Lab Mimic environment config class for Franka put and close door task.
            """

            def __post_init__(self):
                # post init of parents
                super().__post_init__()

                # Override the existing values
                self.datagen_config.name = "put_and_close_door_task_D0"
                self.datagen_config.generation_guarantee = True
                self.datagen_config.generation_keep_failed = False
                self.datagen_config.generation_num_trials = 100
                self.datagen_config.generation_select_src_per_subtask = False
                self.datagen_config.generation_select_src_per_arm = False
                self.datagen_config.generation_relative = False
                self.datagen_config.generation_joint_pos = False
                self.datagen_config.generation_transform_first_robot_pose = False
                self.datagen_config.generation_interpolate_from_last_target_pose = True
                self.datagen_config.max_num_failures = 25
                self.datagen_config.seed = 1

        # Get assets
        embodiment = self.asset_registry.get_asset_by_name("gr1_pink")(enable_cameras=args_cli.enable_cameras)
        kitchen_background = self.asset_registry.get_asset_by_name("lightwheel_robocasa_kitchen")(
            style_id=args_cli.kitchen_style
        )
        vegetable = self.asset_registry.get_asset_by_name(args_cli.vegetable)()
        light = self.asset_registry.get_asset_by_name("light")()

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set initial poses
        embodiment.set_initial_pose(
            Pose(position_xyz=(3.943, -1.04, 0.92), rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068))
        )

        vegetable.set_initial_pose(
            # Bench position
            Pose(position_xyz=(3.922, -0.617, 1.019), rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068))
        )

        # Create refrigerator reference (OpenableObjectReference)
        refrigerator = OpenableObjectReference(
            name="refrigerator",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group",
            parent_asset=kitchen_background,
            openable_joint_name="fridge_door_joint",
            openable_threshold=0.5,
        )

        # Create refrigerator shelf reference (destination for pick and place)
        refrigerator_shelf = ObjectReference(
            name="refrigerator_shelf",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group/Refrigerator034",
            parent_asset=kitchen_background,
        )

        # Create pick and place task
        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=vegetable,
            destination_object=refrigerator,
            destination_location=refrigerator_shelf,
            background_scene=kitchen_background,
        )

        # Create close door task
        close_door_task = CloseDoorTask(
            openable_object=refrigerator,
            closedness_threshold=0.10,
            reset_openness=0.5,
        )

        # Create sequential task
        sequential_task = PutAndCloseDoorTask(subtasks=[pick_and_place_task, close_door_task])

        # Create scene
        scene = Scene(assets=[kitchen_background, vegetable, light, refrigerator, refrigerator_shelf])

        # Create and return environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=sequential_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--vegetable", type=str, default="sweet_potato", help="Type of vegetable to pick and place")
        parser.add_argument(
            "--kitchen_style", type=int, default=2, help="Kitchen style ID for lightwheel robocasa kitchen"
        )
        parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
