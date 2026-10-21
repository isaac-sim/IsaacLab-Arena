Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

On this page we briefly describe the RL environment used in this example workflow
and validate that we can load it in Isaac Lab.

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The environment factory implements ``build(cfg)``, using its typed
``ArenaEnvironmentCfg`` subclass to configure the scene, embodiment, and task.


.. dropdown:: The Lift Object RL Environment
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/lift_object_environment.py
      :language: python
      :start-at: from __future__ import annotations


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset Registry**

.. code-block:: python

   background = self.asset_registry.get_asset_by_name("table")()
   pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
   ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
   light = self.asset_registry.get_asset_by_name("light")()
   assets = [background, pick_up_object, ground_plane, light]

   embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
       enable_cameras=cfg.enable_cameras, concatenate_observation_terms=True
   )
   if cfg.teleop_device is not None:
       teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
   else:
       teleop_device = None

Here, we're selecting the components needed for our RL task: a table as our support surface,
an object to lift (configurable via CLI, default is ``dex_cube``), a ground plane for physics,
and lighting for visualization. The Franka embodiment is configured with ``concatenate_observation_terms=True``
to provide a flat observation vector suitable for RL training.

**2. Position the Objects**

.. code-block:: python

   background.set_initial_pose(Pose(position_xyz=(0.5, 0, 0), rotation_xyzw=(0, 0, 0.707, 0.707)))
   pick_up_object.set_initial_pose(Pose(position_xyz=(0.5, 0, 0.055), rotation_xyzw=(0, 0, 0, 1)))
   ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

Before we create the scene, we need to place our objects in the right locations. The table is positioned
at (0.5, 0, 0), the object is placed on top of the table at a height of 0.055m, and the ground plane
is positioned below to provide physical support.

**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=assets)

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/scene/index` for scene composition details.

**4. Create the Lift Object RL Task**

.. code-block:: python

    task = LiftObjectTaskRL(
        pick_up_object,
        background,
        embodiment,
        minimum_height_to_lift=0.04,
        episode_length_s=5.0,
        rl_training_mode=cfg.rl_training_mode,
    )

The ``LiftObjectTaskRL`` encapsulates the RL training objective: lift the object to commanded target positions.
The task includes:

- **Command Manager**: Samples random target positions within a configurable range
- **Reward Terms**: Dense rewards for reaching, grasping, lifting, and achieving target poses
- **Observation Space**: Robot state (joint positions, velocities), object state (pose, velocity), and goal commands
- **Termination Conditions**: Object dropped or timeout
- **Success Condition**: Object reaches target position (disabled by the ``--rl_training_mode`` flag for training)

See :doc:`../../concepts/task/index` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy
   from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,
       rl_framework_entry_point="rsl_rl_cfg_entry_point",
       rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
       env_cfg_callback=set_control_rate_50hz,
   )

Finally, we assemble all the pieces into a complete, runnable RL environment. The ``IsaacLabArenaEnvironment``
connects the embodiment (the robot), the scene (the world), and the task (the objective and rewards).
See :doc:`../../concepts/environment/index` for environment composition details.


Validation: Run Random Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate the environment loads correctly, run one training iteration and check for errors:

.. code-block:: bash

   python isaaclab_arena/scripts/train.py \
     --rl_library rsl_rl \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task lift_object \
     --rl_training_mode \
     --num_envs 64 \
     --max_iterations 1


If the environment is set up correctly, you will see one iteration of training output before the script exits.

You should see a summary like the following. Timings and metric values vary with hardware,
configuration, and random seed.

.. code-block:: text

   Learning iteration 0/1

                               Total steps: 1536
                          Steps per second: 1343
                           Collection time: 0.981s
                             Learning time: 0.162s
                           Mean value loss: 0.0265
                       Mean surrogate loss: -0.0138
                         Mean entropy loss: 11.3535
                               Mean reward: 0.65
                       Mean episode length: 12.00
                           Mean action std: 1.00
        Episode_Termination/object_dropped: 0.0000
             Episode_Reward/lifting_object: 0.1050
            Episode_Reward/reaching_object: 0.0000
   Episode_Reward/object_goal_tracking_fine_grained: 0.0000
        Metrics/object_pose/position_error: 0.4727
                  Episode_Reward/joint_vel: -0.0001
              Episode_Termination/time_out: 0.0423
                Episode_Reward/action_rate: -0.0001
       Episode_Reward/object_goal_tracking: 0.0224
     Metrics/object_pose/orientation_error: 2.3027
               Episode_Termination/success: 0.0000
   --------------------------------------------------------------------------------
                            Iteration time: 1.14s
                              Time elapsed: 0:00:01
                                       ETA: 0:00:00
