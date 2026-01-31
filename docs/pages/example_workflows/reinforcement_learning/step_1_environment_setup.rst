Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

On this page we briefly describe the RL environment used in this example workflow
and validate that we can load it in Isaac Lab.

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^


.. dropdown:: The Lift Object RL Environment
   :animate: fade-in

   .. code-block:: python

      class LiftObjectEnvironment(ExampleEnvironmentBase):

          name: str = "lift_object"

          def get_env(self, args_cli: argparse.Namespace):
              from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
              from isaaclab_arena.scene.scene import Scene
              from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL
              from isaaclab_arena.utils.pose import Pose

              background = self.asset_registry.get_asset_by_name("table")()
              pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()

              # Add ground plane and light to the scene
              ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
              light = self.asset_registry.get_asset_by_name("light")()

              assets = [background, pick_up_object, ground_plane, light]

              embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                  concatenate_observation_terms=True
              )

              # Set all positions
              background.set_initial_pose(Pose(position_xyz=(0.5, 0, 0), rotation_wxyz=(0.707, 0, 0, 0.707)))
              pick_up_object.set_initial_pose(Pose(position_xyz=(0.5, 0, 0.055), rotation_wxyz=(1, 0, 0, 0)))
              ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

              # Compose the scene
              scene = Scene(assets=assets)

              task = LiftObjectTaskRL(
                  pick_up_object,
                  background,
                  embodiment,
                  minimum_height_to_lift=0.04,
                  episode_length_s=5.0,
                  rl_training_mode=args_cli.rl_training_mode,
              )

              isaaclab_arena_environment = IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=task,
                  teleop_device=None,
              )

              return isaaclab_arena_environment


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset Registry**

.. code-block:: python

   background = self.asset_registry.get_asset_by_name("table")()
   pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
   ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
   light = self.asset_registry.get_asset_by_name("light")()

   embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
       concatenate_observation_terms=True
   )

Here, we're selecting the components needed for our RL task: a table as our support surface,
an object to lift (configurable via CLI, default is ``dex_cube``), a ground plane for physics,
and lighting for visualization. The Franka embodiment is configured with ``concatenate_observation_terms=True``
to provide a flat observation vector suitable for RL training.

**2. Position the Objects**

.. code-block:: python

   background.set_initial_pose(Pose(position_xyz=(0.5, 0, 0), rotation_wxyz=(0.707, 0, 0, 0.707)))
   pick_up_object.set_initial_pose(Pose(position_xyz=(0.5, 0, 0.055), rotation_wxyz=(1, 0, 0, 0)))
   ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

Before we create the scene, we need to place our objects in the right locations. The table is positioned
at (0.5, 0, 0), the object is placed on top of the table at a height of 0.055m, and the ground plane
is positioned below to provide physical support.

**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=assets)

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/concept_scene_design` for scene composition details.

**4. Create the Lift Object RL Task**

.. code-block:: python

    task = LiftObjectTaskRL(
        pick_up_object,
        background,
        embodiment,
        minimum_height_to_lift=0.04,
        episode_length_s=5.0,
        rl_training_mode=args_cli.rl_training_mode,
    )

The ``LiftObjectTaskRL`` encapsulates the RL training objective: lift the object to commanded target positions.
The task includes:

- **Command Manager**: Samples random target positions within a configurable range
- **Reward Terms**: Dense rewards for reaching, grasping, lifting, and achieving target poses
- **Observation Space**: Robot state (joint positions, velocities), object state (pose, velocity), and goal commands
- **Termination Conditions**: Object dropped or timeout
- **Success Condition**: Object reaches target position (only active during evaluation when ``rl_training_mode=False``)

See :doc:`../../concepts/concept_tasks_design` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=None,
   )

Finally, we assemble all the pieces into a complete, runnable RL environment. The ``IsaacLabArenaEnvironment``
connects the embodiment (the robot), the scene (the world), and the task (the objective and rewards).
See :doc:`../../concepts/concept_environment_design` for environment composition details.


Validation: Run Random Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate the environment setup, we can run a random policy to ensure everything loads correctly:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train.py \
     --num_envs 64 \
     --max_iterations 1 \
     lift_object

This command will load the environment, initialize 64 parallel environments, and exit immediately
(``max_iterations=1``). If successful, the environment is ready for training.

You should see output indicating the start of training:

.. code-block:: text

   Learning iteration 0/1

                       Computation: 308 steps/s (collection: 4.600s, learning 0.377s)
             Mean action noise std: 1.00
          Mean value_function loss: 0.0273
               Mean surrogate loss: -0.0138
                 Mean entropy loss: 9.9339
                       Mean reward: 0.65
               Mean episode length: 12.00
        Episode_Reward/action_rate: -0.0000
          Episode_Reward/joint_vel: -0.0001
    Episode_Reward/reaching_object: 0.0000
     Episode_Reward/lifting_object: 0.1050
Episode_Reward/object_goal_tracking: 0.0223
Episode_Reward/object_goal_tracking_fine_grained: 0.0000
Metrics/object_pose/position_error: 0.5721
Metrics/object_pose/orientation_error: 2.2834
      Episode_Termination/time_out: 0.0423
Episode_Termination/object_dropped: 0.0000
       Episode_Termination/success: 0.0000
--------------------------------------------------------------------------------
                   Total timesteps: 1536
                    Iteration time: 4.98s
                      Time elapsed: 00:00:04
                               ETA: 00:00:49
