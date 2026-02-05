Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^


.. dropdown:: The GR1 Sequential Pick & Place and Close Door Environment
   :animate: fade-in

   .. code-block:: python

      class GR1PutAndCloseDoorEnvironment(ExampleEnvironmentBase):

          name: str = "put_item_in_fridge_and_close_door"

          def get_env(self, args_cli: argparse.Namespace):
            from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
            from isaaclab.utils import configclass

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
            from isaaclab_arena.utils.pose import Pose, PoseRange

            def get_pose_range(z_position, yaw):
                return PoseRange(
                    position_xyz_min=(
                        4.05 - RANDOMIZATION_HALF_RANGE_X_M,
                        -0.58 - RANDOMIZATION_HALF_RANGE_Y_M,
                        z_position - RANDOMIZATION_HALF_RANGE_Z_M,
                    ),
                    position_xyz_max=(
                        4.05 + RANDOMIZATION_HALF_RANGE_X_M,
                        -0.58 + RANDOMIZATION_HALF_RANGE_Y_M,
                        z_position + RANDOMIZATION_HALF_RANGE_Z_M,
                    ),
                    rpy_min=(0.0, 0.0, yaw),
                    rpy_max=(0.0, 0.0, yaw),
                )

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

                    # Set custom values for Mimic subtask term offset range and action noise
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

            camera_offset = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_wxyz=(0.57469, 0.11204, -0.17712, -0.79108))
            # Get assets
            embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                enable_cameras=args_cli.enable_cameras, camera_offset=camera_offset
            )
            kitchen_background = self.asset_registry.get_asset_by_name("lightwheel_robocasa_kitchen")(
                style_id=args_cli.kitchen_style
            )

            kitchen_counter_top = ObjectReference(
                name="kitchen_counter_top",
                prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/counter_right_main_group/top_geometry",
                parent_asset=kitchen_background,
            )
            kitchen_counter_top.add_relation(IsAnchor())

            pickup_object = self.asset_registry.get_asset_by_name(args_cli.object)()
            light = self.asset_registry.get_asset_by_name("light")()

            if args_cli.teleop_device is not None:
                teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
            else:
                teleop_device = None

            # Set initial poses
            embodiment.set_initial_pose(
                Pose(
                    position_xyz=(3.943, -1.0, 0.995),
                    rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068),
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

            # Consider changing to other values for different objects, below is for ranch dressing bottle
            z_position = 1.0082
            yaw_rad = math.radians(-111.55)
            assert args_cli.object_set is None, "Object set is not supported yet"
            #  All obs from object set are under the same randomization range
            if args_cli.object_set is not None and len(args_cli.object_set) > 0:
                objects = []
                for obj in args_cli.object_set:
                    obj_from_set = self.asset_registry.get_asset_by_name(obj)()
                    objects.append(obj_from_set)
                object_set = RigidObjectSet(name="object_set", objects=objects)
                object_set.set_initial_pose(get_pose_range(z_position, yaw_rad))
                # Create scene
                scene = Scene(assets=[kitchen_background, object_set, light, refrigerator, refrigerator_shelf])
            else:
                pickup_object.add_relation(On(kitchen_counter_top))
                # Place the object at a specific position GR1 to be able to reach it with its hand.
                pickup_object.add_relation(AtPosition(x=4.05, y=-0.58))
                pickup_object.add_relation(RotateAroundSolution(yaw_rad=yaw_rad))
                pickup_object.add_relation(
                    RandomAroundSolution(
                        x_half_m=RANDOMIZATION_HALF_RANGE_X_M,
                        y_half_m=RANDOMIZATION_HALF_RANGE_Y_M,
                        z_half_m=RANDOMIZATION_HALF_RANGE_Z_M,
                    )
                )
                # Create scene
                scene = Scene(
                    assets=[kitchen_background, kitchen_counter_top, pickup_object, light, refrigerator, refrigerator_shelf]
                )

            # Create pick and place task
            pick_and_place_task = PickAndPlaceTask(
                pick_up_object=pickup_object if args_cli.object_set is None else object_set,
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



Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. code-block:: python

    camera_offset = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_wxyz=(0.57469, 0.11204, -0.17712, -0.79108))
    embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras, camera_offset=camera_offset)
    kitchen_background = self.asset_registry.get_asset_by_name("lightwheel_robocasa_kitchen")(style_id=args_cli.kitchen_style)
    kitchen_counter_top = ObjectReference(
        name="kitchen_counter_top",
        prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/counter_right_main_group/top_geometry",
        parent_asset=kitchen_background,
    )
    kitchen_counter_top.add_relation(IsAnchor())

    pickup_object = self.asset_registry.get_asset_by_name(args_cli.object)()
    light = self.asset_registry.get_asset_by_name("light")()

    if args_cli.teleop_device is not None:
        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
    else:
        teleop_device = None

Here, we're selecting the components needed for our sequential static manipulation task: 
The GR1 embodiment, the kitchen environment as our background, the object to pick and place,
and a light to illuminate the scene.
The ``AssetRegistry`` and ``DeviceRegistry`` have been initialized in the ``ExampleEnvironmentBase`` class.
See :doc:`../../concepts/concept_assets_design` for details on asset architecture.


**2. Position the Embodiment and Objects**

.. code-block:: python

    # Set initial poses
    embodiment.set_initial_pose(
        Pose(
            position_xyz=(3.943, -1.0, 0.995),
            rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068),
        )
    )

    # ...

    # Consider changing to other values for different objects, below is for ranch dressing bottle
    z_position = 1.0082
    yaw_rad = math.radians(-111.55)
    assert args_cli.object_set is None, "Object set is not supported yet"
    #  All obs from object set are under the same randomization range
    if args_cli.object_set is not None and len(args_cli.object_set) > 0:
        objects = []
        for obj in args_cli.object_set:
            obj_from_set = self.asset_registry.get_asset_by_name(obj)()
            objects.append(obj_from_set)
        object_set = RigidObjectSet(name="object_set", objects=objects)
        object_set.set_initial_pose(get_pose_range(z_position, yaw_rad))
        # Create scene
        scene = Scene(assets=[kitchen_background, object_set, light, refrigerator, refrigerator_shelf])
    else:
        pickup_object.add_relation(On(kitchen_counter_top))
        # Place the object at a specific position GR1 to be able to reach it with its hand.
        pickup_object.add_relation(AtPosition(x=4.05, y=-0.58))
        pickup_object.add_relation(RotateAroundSolution(yaw_rad=yaw_rad))
        pickup_object.add_relation(
            RandomAroundSolution(
                x_half_m=RANDOMIZATION_HALF_RANGE_X_M,
                y_half_m=RANDOMIZATION_HALF_RANGE_Y_M,
                z_half_m=RANDOMIZATION_HALF_RANGE_Z_M,
            )
        )
        # Create scene
        scene = Scene(
            assets=[kitchen_background, kitchen_counter_top, pickup_object, light, refrigerator, refrigerator_shelf]
        )

Before we create the scene, we need to place our embodiment and objects in the right locations.
The embodiment is placed in a fixed spot while the object is placed on top of the kitchen counter using
the relational object placement APIs. The object is placed within a randomzation range to add some variability to the task.


**3. Create the Sequential Pick & Place and Close Door Task**

.. code-block:: python

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
    sequential_task = PutAndCloseDoorTask(subtasks=[pick_and_place_task, close_door_task])

The sequential task is composed of two atomic subtasks: the pick and place task and the close door task.
See :doc:`../../concepts/concept_tasks_design` for task creation details.


**4. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=[kitchen_background, pickup_object, light, refrigerator, refrigerator_shelf])

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/concept_scene_design` for scene composition details.


**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name=self.name,
        embodiment=embodiment,
        scene=scene,
        task=sequential_task,
        teleop_device=teleop_device,
    )

Finally, we assemble all the pieces into a complete, runnable environment. The ``IsaacLabArenaEnvironment`` is the
top-level container that connects the embodiment (the robot), the scene (the world), and the task (the objective).
See :doc:`../../concepts/concept_environment_design` for environment composition details.


Step 1: Download a Test Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run a robot in the environment we need some recorded demonstration data that
can be fed to the robot to control its actions.

TODO: ADD INSTRUCTIONS TO DOWNLOAD A TEST DATASET


Step 2: Validate the Environment by Replaying the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the downloaded dataset to verify the environment setup:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file "${DATASET_DIR}/arena_gr1_sequential_manipulation_dataset_generated.hdf5" \
     put_item_in_fridge_and_close_door \
     --object ranch_dressing_bottle \
     --embodiment gr1_pink

You should see the GR1 robot replaying the demonstrations, performing the sequential
pick & place and close door task in the kitchen environment.

.. figure:: ../../../images/gr1_sequential_static_manipulation_env.gif
   :width: 100%
   :alt: GR1 picking up and placing an object in a refrigerator and closing the door
   :align: center

   IsaacLab Arena GR1 picking up and placing an object in a refrigerator and closing the door
