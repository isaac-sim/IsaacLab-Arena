# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


# Nominal position for the target object only. Distractors have no AtPosition; they only get On(counter) and NoCollision.
TARGET_OBJECT_X = 4.05
TARGET_OBJECT_Y = -0.58

class GR1MultiObjectPutAndCloseDoorEnvironment(ExampleEnvironmentBase):
    """
    Multi-object variant of put-and-close-door: multiple objects on the kitchen counter,
    one target object to pick and place into the fridge, then close the door.
    The target (ranch bottle by default) is fixed at (TARGET_OBJECT_X, TARGET_OBJECT_Y)
    on the counter and marked as an anchor so it never falls; the relation solver only
    places distractors with On(counter) and NoCollision. Only the target (--object)
    is used for the pick-and-place task.
    """

    name = "gr1_multi_object_put_and_close_door"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
        from isaaclab.utils import configclass

        from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import (
            IsAnchor,
            NoCollision,
            On,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
        from isaaclab_arena.tasks.task_base import TaskBase
        from isaaclab_arena.utils.pose import Pose

        class PutAndCloseDoorTask(SequentialTaskBase):
            def __init__(
                self,
                subtasks: list[TaskBase],
                episode_length_s: float | None = None,
            ):
                super().__init__(
                    subtasks=subtasks, episode_length_s=episode_length_s, desired_subtask_success_state=[True, True]
                )

            def get_viewer_cfg(self):
                return self.subtasks[0].get_viewer_cfg()

            def get_prompt(self):
                return None

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                mimic_env_cfg = PutAndCloseDoorTaskMimicEnvCfg()
                mimic_env_cfg.subtask_configs = self.combine_mimic_subtask_configs(ArmMode.RIGHT)
                for eef_name, subtask_list in mimic_env_cfg.subtask_configs.items():
                    for subtask_config in subtask_list:
                        subtask_config.subtask_term_offset_range = (0, 0)
                        subtask_config.action_noise = 0.003
                return mimic_env_cfg

        @configclass
        class PutAndCloseDoorTaskMimicEnvCfg(MimicEnvCfg):
            def __post_init__(self):
                super().__post_init__()
                self.datagen_config.name = "gr1_multi_object_put_and_close_door_D0"
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(self.datagen_config, key, value)

        camera_offset = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_wxyz=(0.57469, 0.11204, -0.17712, -0.79108))
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras, camera_offset=camera_offset
        )
        kitchen_background = self.asset_registry.get_asset_by_name("lightwheel_robocasa_kitchen")(
            style_id=args_cli.kitchen_style
        )

        # Kitchen "top" = top surface of the right counter (counter_right_main_group).
        kitchen_counter_top = ObjectReference(
            name="kitchen_counter_top",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/counter_right_main_group/top_geometry",
            parent_asset=kitchen_background,
        )
        kitchen_counter_top.add_relation(IsAnchor())

        # Fixed counter accessories (knife block, toaster) for NoCollision; same style as refrigerator below.
        knife_block_ref = ObjectReference(
            name="knife_block",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/knife_block_main_group",
            parent_asset=kitchen_background,
        )
        knife_block_ref.add_relation(IsAnchor())
        toaster_ref = ObjectReference(
            name="toaster",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/toaster_main_group",
            parent_asset=kitchen_background,
        )
        toaster_ref.add_relation(IsAnchor())
        counter_accessory_refs = [knife_block_ref, toaster_ref]

        light = self.asset_registry.get_asset_by_name("light")()

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        embodiment.set_initial_pose(
            Pose(
                position_xyz=(3.943, -1.0, 0.995),
                rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068),
            )
        )

        refrigerator = OpenableObjectReference(
            name="refrigerator",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group",
            parent_asset=kitchen_background,
            openable_joint_name="fridge_door_joint",
            openable_threshold=0.5,
        )

        refrigerator_shelf = ObjectReference(
            name="refrigerator_shelf",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group/Refrigerator034",
            parent_asset=kitchen_background,
        )

        # Target object: fixed on the counter at (TARGET_OBJECT_X, TARGET_OBJECT_Y) so it does not fall.
        # We set its initial pose and mark it IsAnchor() so the placer never randomizes or overwrites it;
        # only distractors are optimized by the relation solver.
        target_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        counter_top_z = kitchen_counter_top.get_world_bounding_box().max_point[2]
        target_bbox = target_object.get_bounding_box()
        clearance = 0.01
        target_z = counter_top_z + clearance - target_bbox.min_point[2]
        yaw_rad = math.radians(-111.55)
        yaw_quat_wxyz = (math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))
        target_object.set_initial_pose(
            Pose(
                position_xyz=(TARGET_OBJECT_X, TARGET_OBJECT_Y, target_z),
                rotation_wxyz=yaw_quat_wxyz,
            )
        )
        target_object.add_relation(IsAnchor())
        target_object.add_relation(On(kitchen_counter_top))
        for ref in counter_accessory_refs:
            target_object.add_relation(NoCollision(ref))

        # Distractor objects: On(counter) + NoCollision only (no AtPosition). Solver places them on the counter without overlapping target, knife block, toaster, or each other.
        distractor_objects = []
        for name in args_cli.distractor_objects or []:
            obj = self.asset_registry.get_asset_by_name(name)()
            obj.add_relation(On(kitchen_counter_top))
            for ref in counter_accessory_refs:
                obj.add_relation(NoCollision(ref))
            target_object.add_relation(NoCollision(obj))
            distractor_objects.append(obj)
        for i, obj_a in enumerate(distractor_objects):
            for obj_b in distractor_objects[i + 1 :]:
                obj_a.add_relation(NoCollision(obj_b))

        scene = Scene(
            assets=[
                kitchen_background,
                kitchen_counter_top,
                *counter_accessory_refs,
                target_object,
                *distractor_objects,
                light,
                refrigerator,
                refrigerator_shelf,
            ]
        )

        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=target_object,
            destination_object=refrigerator,
            destination_location=refrigerator_shelf,
            background_scene=kitchen_background,
        )

        close_door_task = CloseDoorTask(
            openable_object=refrigerator,
            closedness_threshold=0.10,
            reset_openness=0.5,
        )

        sequential_task = PutAndCloseDoorTask(subtasks=[pick_and_place_task, close_door_task], episode_length_s=10.0)

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
        parser.add_argument(
            "--object",
            type=str,
            default="ranch_dressing_bottle",
            choices=[
                "sweet_potato",
                "jug",
                "ranch_dressing_bottle",
                "tomato_soup_can",
                "cracker_box",
            ],
            help="Target object to pick and place into the fridge.",
        )
        parser.add_argument(
            "--distractor_objects",
            nargs="*",
            type=str,
            default=None,
            help=(
                "Optional object names on the kitchen counter as distractors. "
                "Relation solver places all with NoCollision; only --object is the pick-and-place target."
            ),
        )
        parser.add_argument(
            "--kitchen_style", type=int, default=2, help="Kitchen style ID for lightwheel robocasa kitchen"
        )
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
        parser.add_argument("--embodiment", type=str, default="gr1_pink", help="Robot embodiment to use")
