Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^


.. dropdown:: The Franka Lift Object Environment
   :animate: fade-in

   .. code-block:: python

      class LiftObjectEnvironment(ExampleEnvironmentBase):

        name: str = "lift_object"

        def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
            from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
            from isaaclab_arena.scene.scene import Scene
            from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL
            from isaaclab_arena.utils.pose import Pose, PoseRange

            background = self.asset_registry.get_asset_by_name("table")()
            pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()

            # Add ground plane and light to the scene
            ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
            light = self.asset_registry.get_asset_by_name("light")()

            assets = [background, pick_up_object, ground_plane, light]

            embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(concatenate_observation_terms=True)

            if args_cli.teleop_device is not None:
                teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
            else:
                teleop_device = None

            # Set all positions
            background.set_initial_pose(Pose(position_xyz=(0.5, 0, 0), rotation_wxyz=(0.707, 0, 0, 0.707)))
            pick_up_object.set_initial_pose(Pose(position_xyz=(0.5, 0, 0.055), rotation_wxyz=(1, 0, 0, 0)))
            reset_pose_range = PoseRange(position_xyz_min=(-0.1, -0.25, 0.0), position_xyz_max=(0.1, 0.25, 0.0))
            ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

            # Compose the scene
            scene = Scene(assets=assets)

            task = LiftObjectTaskRL(
                pick_up_object,
                background,
                embodiment,
                minimum_height_to_lift=0.04,
                episode_length_s=5.0,
                reset_pose_range=reset_pose_range,
            )

            isaaclab_arena_environment = IsaacLabArenaEnvironment(
                name=self.name,
                embodiment=embodiment,
                scene=scene,
                task=task,
                teleop_device=teleop_device,
            )

            return isaaclab_arena_environment


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset and Device Registry**

.. code-block:: python

   background = self.asset_registry.get_asset_by_name("table")()
   pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()

   ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
   light = self.asset_registry.get_asset_by_name("light")()

   assets = [background, pick_up_object, ground_plane, light]

   embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(concatenate_observation_terms=True)

   if args_cli.teleop_device is not None:
       teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
   else:
       teleop_device = None

Here, we're selecting the components needed for our lift object task: a table as our background surface,
an object to be lifted (configurable via command line), and the Franka robot arm as our embodiment.
We also add a ground plane and lighting for proper rendering.
The ``AssetRegistry`` and ``DeviceRegistry`` have been initialized in the ``ExampleEnvironmentBase`` class.
We do not use the device registry for the RL example workflow.
See :doc:`../../concepts/concept_assets_design` for details on asset architecture.

**2. Position the Objects**

.. code-block:: python

   background.set_initial_pose(Pose(position_xyz=(0.5, 0, 0), rotation_wxyz=(0.707, 0, 0, 0.707)))
   pick_up_object.set_initial_pose(Pose(position_xyz=(0.5, 0, 0.055), rotation_wxyz=(1, 0, 0, 0)))
   reset_pose_range = PoseRange(position_xyz_min=(-0.1, -0.25, 0.0), position_xyz_max=(0.1, 0.25, 0.0))
   ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

Before we create the scene, we need to place our objects in the right locations. The table is positioned
in front of the robot, and the object to be lifted is placed on top of the table surface. We also define
a ``reset_pose_range`` that specifies how the object's position can vary relative to its initial pose
during episode resets, adding variability for reinforcement learning training.


**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=assets)

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/concept_scene_design` for scene composition details.

**4. Create the Lift Object Task**

.. code-block:: python

    task = LiftObjectTaskRL(
        pick_up_object,
        background,
        embodiment,
        minimum_height_to_lift=0.04,
        episode_length_s=5.0,
        reset_pose_range=reset_pose_range,
    )

The ``LiftObjectTaskRL`` encapsulates the goal of this environment: lift the object to a specified height.
The task includes parameters like the minimum height threshold (0.04 meters) and episode duration (5 seconds).
The RL task is a subclass of the ``LiftObjectTask`` task.
It includes additional reward terms for the reinforcement learning training.
See :doc:`../../concepts/concept_tasks_design` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,
   )

Finally, we assemble all the pieces into a complete, runnable environment. The ``IsaacLabArenaEnvironment`` is the
top-level container that connects the embodiment (the robot arm), the scene (the table and object), and the task (the lifting objective).
See :doc:`../../concepts/concept_environment_design` for environment composition details.


Validate the Environment
^^^^^^^^^^^^^^^^^^^^^^^^

To validate that the environment is set up correctly, you can run a simple test to visualize the scene
and verify that all components are properly loaded. The lift object environment is designed for reinforcement
learning training, so validation typically involves checking that the robot can interact with the object
and that the reward signal is computed correctly.

You can test the environment by running:

.. code-block:: bash

   python isaaclab_arena/policy/zero_action_policy.py lift_object \
     --embodiment franka \
     --object dex_cube \
     --num_envs 1

This will launch a single environment instance where you can observe the Franka robot arm positioned
near the table with the cube object ready to be lifted.

.. note::

   For reinforcement learning training with this environment, see the next steps in this workflow where
   we configure and run the RL training process.
