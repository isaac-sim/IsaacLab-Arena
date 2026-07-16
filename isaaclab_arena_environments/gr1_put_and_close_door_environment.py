# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


RANDOMIZATION_HALF_RANGE_X_M = 0.03
RANDOMIZATION_HALF_RANGE_Y_M = 0.01
RANDOMIZATION_HALF_RANGE_Z_M = 0.0

# The GR1 embodiments that are compatible with this environment.
GR1_EMBODIMENTS = ("gr1_joint", "gr1_pink")


@dataclass
class GR1PutAndCloseDoorEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1 put-and-close-door environment."""

    enable_cameras: bool = False
    object: str = "ranch_dressing_hope_robolab"
    object_set: list[str] | None = None
    kitchen_style: int = 2
    teleop_device: str | None = None
    embodiment: str = "gr1_pink"


@register_environment
class GR1PutAndCloseDoorEnvironment(ArenaEnvironmentFactory[GR1PutAndCloseDoorEnvironmentCfg]):
    """
    A sequential task environment with two subtasks for GR1 humanoid robot:
    1. Pick and place object into the refrigerator shelf
    2. Close the refrigerator door
    The refrigerator starts open, the robot places the object inside, then closes it.
    Uses the lightwheel robocasa kitchen background.
    """

    name = "put_item_in_fridge_and_close_door"
    _legacy_argparse_cfg_type = GR1PutAndCloseDoorEnvironmentCfg

    def build(self, cfg: GR1PutAndCloseDoorEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils.configclass import configclass

        from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import (
            AtPosition,
            IsAnchor,
            On,
            RandomAroundSolution,
            RotateAroundSolution,
        )
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

                # Override default subtask term offset range and action noise
                for eef_name, subtask_list in mimic_env_cfg.subtask_configs.items():
                    for subtask_config in subtask_list:
                        subtask_config.subtask_term_offset_range = (0, 0)
                        subtask_config.action_noise = 0.003

                return mimic_env_cfg

        @configclass
        class PutAndCloseDoorTaskMimicEnvCfg(MimicEnvCfg):
            """
            Isaac Lab Mimic environment config class for GR1 put and close door task.
            """

            def __post_init__(self):
                # post init of parents
                super().__post_init__()

                # Override the existing values
                self.datagen_config.name = "put_and_close_door_task_D0"
                # Use default mimic datagen config parameters
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(self.datagen_config, key, value)

        camera_offset = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_xyzw=(0.11204, -0.17712, -0.79108, 0.57469))
        assert (
            cfg.embodiment in GR1_EMBODIMENTS
        ), f"{self.name} only supports GR1 embodiments {GR1_EMBODIMENTS}, got '{cfg.embodiment}'."
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.camera_config.robot_pov_cam.offset = CameraCfg.OffsetCfg(
            pos=camera_offset.position_xyz,
            rot=camera_offset.rotation_xyzw,
            convention="opengl",
        )
        kitchen_background = self.asset_registry.get_asset_by_name("lightwheel_robocasa_kitchen")(
            style_id=cfg.kitchen_style
        )

        kitchen_counter_top = ObjectReference(
            name="kitchen_counter_top",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/counter_right_main_group/top_geometry",
            parent_asset=kitchen_background,
        )
        kitchen_counter_top.add_relation(IsAnchor())

        light = self.asset_registry.get_asset_by_name("light")()

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Set initial poses
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(3.943, -1.0, 0.995),
                rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
            )
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

        if cfg.object_set is not None and len(cfg.object_set) > 0:
            objects = [self.asset_registry.get_asset_by_name(obj)() for obj in cfg.object_set]
            pickup_object = RigidObjectSet(name="object_set", objects=objects)
        else:
            pickup_object = self.asset_registry.get_asset_by_name(cfg.object)()

        pickup_object.add_relation(On(kitchen_counter_top))
        pickup_object.add_relation(AtPosition(x=4.05, y=-0.58))
        # Consider changing to other values for different objects, below is for ranch dressing bottle.
        yaw_rad = math.radians(-111.55)
        pickup_object.add_relation(RotateAroundSolution(yaw_rad=yaw_rad))
        pickup_object.add_relation(
            RandomAroundSolution(x_half_m=RANDOMIZATION_HALF_RANGE_X_M, y_half_m=RANDOMIZATION_HALF_RANGE_Y_M)
        )
        scene = Scene(
            assets=[kitchen_background, kitchen_counter_top, pickup_object, light, refrigerator, refrigerator_shelf]
        )

        # Create pick and place task
        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=pickup_object,
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
        sequential_task = PutAndCloseDoorTask(subtasks=[pick_and_place_task, close_door_task], episode_length_s=10.0)

        # Create and return environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=sequential_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment
