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


class FrankaPutAndCloseContainerEnvironment(ExampleEnvironmentBase):
    """
    A sequential task environment with two subtasks:
    1. Pick and place object into a container (cabinet drawer or microwave)
    2. Close the container door
    The container starts open, the robot places the object inside, then closes it.
    Supports both cabinet and microwave containers via --container CLI argument.
    """

    name = "franka_put_and_close_container"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
        from isaaclab_arena.utils.pose import Pose

        pick_object = self.asset_registry.get_asset_by_name(args_cli.object)()

        scene_assets = [pick_object]

        # Get embodiment based on selection
        if args_cli.embodiment == "franka":
            embodiment = self.asset_registry.get_asset_by_name("franka")(enable_cameras=args_cli.enable_cameras)
        else:
            raise NotImplementedError(f"Embodiment {args_cli.embodiment} not supported")

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Configure based on container type
        if args_cli.container == "cabinet":
            task_description_pick = "Pick the object from the top of the cabinet and place it in the drawer."
            task_description_close = "Close the cabinet drawer."

            # Create a simple background object with object_min_z
            class SimpleBackground:
                def __init__(self, object_min_z: float):
                    self.object_min_z = object_min_z
            background = SimpleBackground(object_min_z=0.05)

            ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
            light = self.asset_registry.get_asset_by_name("light")()
            # Cabinet scenario
            container = self.asset_registry.get_asset_by_name("cabinet")()
            container.set_initial_pose(Pose(position_xyz=(0.6, 0.0, 0.4), rotation_wxyz=(0.0, 0.0, 0.0, 1.0)))

            # Create object reference to the drawer bottom as destination
            destination_ref = ObjectReference(
                parent_asset=container,
                name="drawer_bottom",
                prim_path="{ENV_REGEX_NS}/cabinet/cabinet/drawer_bottom",
                object_type=ObjectType.RIGID,
            )

            # Get the pick-up object (place it on top of the cabinet)
            pick_object.set_initial_pose(
                Pose(
                    position_xyz=(0.35, 0.0, 1.05),  # On top of cabinet
                    rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                )
            )

            # Set robot initial pose
            embodiment.set_initial_pose(
                Pose(
                    position_xyz=(0.0, 0.0, 0.0),
                    rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                )
            )
            scene_assets = scene_assets + [container, ground_plane, light]

        elif args_cli.container == "microwave":
            task_description_pick = "Pick the object and place it on the microwave disc."
            task_description_close = "Close the microwave door."

            # Microwave scenario - use kitchen background
            background = self.asset_registry.get_asset_by_name("kitchen")()

            container = self.asset_registry.get_asset_by_name("microwave")()
            # Place microwave on the packing table in kitchen (similar to GR1 environment)
            container.set_initial_pose(
                Pose(
                    position_xyz=(0.4, -0.00586, 0.22773),
                    rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
                )
            )

            # Create ObjectReference to the microwave's interior disc as destination
            destination_ref = ObjectReference(
                name="microwave_disc",
                parent_asset=container,
                prim_path="{ENV_REGEX_NS}/microwave/Microwave039_Disc001",
                object_type=ObjectType.RIGID,
            )

            # Get the pick-up object (place it on the kitchen counter)
            pick_object.set_initial_pose(
                Pose(
                    position_xyz=(0.2, -0.437, 0.154),
                    rotation_wxyz=(0.5, -0.5, 0.5, -0.5),
                )
            )

            # Set Franka arm pose for kitchen setup
            embodiment.event_config.init_franka_arm_pose.params["default_pose"] = [
                0.0, -1.309, 0.0, -2.793, 0.0, 3.037, 0.740, 0.04, 0.04
            ]
            embodiment.set_initial_pose(
                Pose(
                    position_xyz=(-0.3, 0.0, -0.5),
                    rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                )
            )

            scene_assets = scene_assets + [background, container]

        else:
            raise ValueError(f"Unsupported container type: {args_cli.container}. Must be 'cabinet' or 'microwave'.")

        # Create scene
        scene = Scene(assets=scene_assets)

        # Create close door task
        close_door_task = CloseDoorTask(
            openable_object=container,
            closedness_threshold=0.05,
            reset_openness=0.3 if args_cli.container == "cabinet" else 0.9,
            task_description=task_description_close,
        )

        # Create pick and place task
        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=pick_object,
            destination_location=destination_ref,
            background_scene=background,
            task_description=task_description_pick,
        )

        # Create a sequential task wrapper class
        class SequentialPutAndCloseDoorTask(SequentialTaskBase):
            def __init__(self, subtasks, episode_length_s=None):
                super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

            def get_metrics(self):
                return []

            def get_mimic_env_cfg(self, arm_mode):
                return None

            def get_viewer_cfg(self):
                return self.subtasks[1].get_viewer_cfg()

        # Create the sequential task
        sequential_task = SequentialPutAndCloseDoorTask(
            subtasks=[pick_and_place_task, close_door_task],
            episode_length_s=90.0,  # Episode for two subtasks
        )

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
            "--container",
            type=str,
            default="cabinet",
            choices=["cabinet", "microwave"],
            help="Container type: cabinet or microwave",
        )
        parser.add_argument("--object", type=str, default="dex_cube", help="Object to pick and place in the container")
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment to use")
        parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
