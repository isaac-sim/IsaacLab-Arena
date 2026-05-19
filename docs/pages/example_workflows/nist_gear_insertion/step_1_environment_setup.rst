Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

On this page we briefly describe the RL environment used in this example workflow
and validate that we can load it in Isaac Lab.

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^


.. dropdown:: The NIST Gear Insertion Environment
   :animate: fade-in

   .. code-block:: python

      class NISTAssembledGearMeshOSCEnvironment(ExampleEnvironmentBase):

          name: str = "nist_assembled_gear_mesh_osc"

          def get_env(self, args_cli: argparse.Namespace):
              import isaaclab.sim as sim_utils

              import isaaclab_arena_environments.mdp as mdp
              from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
              from isaaclab_arena.scene.scene import Scene
              from isaaclab_arena.tasks.nist_gear_insertion.task import (
                  GearInsertionGeometryCfg,
                  NistGearInsertionRLTask,
              )
              from isaaclab_arena.utils.pose import Pose
              from isaaclab_arena_environments.mdp.nist_gear_insertion.franka_osc_cfg import (
                  FrankaNistGearInsertionObservationsCfg,
                  FrankaNistGearInsertionOscActionsCfg,
              )
              from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_rewards import (
                  NistGearInsertionOscRewardsCfg,
              )

              peg_tip_offset = (0.02025, 0.0, 0.025)
              peg_base_offset = (0.02025, 0.0, 0.0)

              table = self.asset_registry.get_asset_by_name("table")()
              assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
              gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
              medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
              light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
              light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

              embodiment = self.asset_registry.get_asset_by_name("franka_nist_gear_insertion_osc")(
                  enable_cameras=args_cli.enable_cameras,
                  concatenate_observation_terms=True,
              )
              embodiment.action_config = FrankaNistGearInsertionOscActionsCfg(
                  fixed_asset_name=gears_and_base.name,
                  peg_offset=peg_tip_offset,
              )
              embodiment.observation_config = FrankaNistGearInsertionObservationsCfg(
                  fixed_asset_name=gears_and_base.name,
                  peg_offset=peg_tip_offset,
                  fingertip_body_name=embodiment.get_command_body_name(),
                  concatenate_observation_terms=embodiment.concatenate_observation_terms,
              )
              embodiment.reward_config = NistGearInsertionOscRewardsCfg(
                  gear_name=medium_gear.name,
                  board_name=gears_and_base.name,
                  peg_offset=peg_base_offset,
                  held_gear_base_offset=peg_base_offset,
                  gear_peg_height=0.02,
                  success_z_fraction=0.20,
                  xy_threshold=0.0025,
              )
              if args_cli.teleop_device is not None:
                  teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
              else:
                  teleop_device = None

              table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
              assembled_board.set_initial_pose(
                  Pose(position_xyz=(0.88, 0.15, -0.009), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
              )
              medium_gear.set_initial_pose(
                  Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
              )
              gears_and_base.set_initial_pose(
                  Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
              )
              scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

              geometry_cfg = GearInsertionGeometryCfg(
                  peg_offset_from_board=list(peg_base_offset),
                  peg_offset_for_obs=list(peg_tip_offset),
                  success_z_fraction=0.20,
                  xy_threshold=0.0025,
              )

              task = NistGearInsertionRLTask(
                  assembled_board=assembled_board,
                  held_gear=medium_gear,
                  background_scene=table,
                  gear_base_asset=gears_and_base,
                  geometry_cfg=geometry_cfg,
                  episode_length_s=15.0,
                  grasp_cfg=embodiment.get_gear_insertion_grasp_config(),
                  fingertip_body_name=embodiment.get_command_body_name(),
                  enable_randomization=True,
                  disable_success_termination=args_cli.disable_success_termination,
              )

              isaaclab_arena_environment = IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=task,
                  teleop_device=teleop_device,
                  env_cfg_callback=mdp.assembly_env_cfg_callback,
              )

              return isaaclab_arena_environment


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset Registry**

.. code-block:: python

   table = self.asset_registry.get_asset_by_name("table")()
   assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
   gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
   medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
   light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
   light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

   embodiment = self.asset_registry.get_asset_by_name("franka_nist_gear_insertion_osc")(
       enable_cameras=args_cli.enable_cameras,
       concatenate_observation_terms=True,
   )

Here, we're selecting the components needed for our RL task: a table as our support surface,
the assembled NIST board for context, the fixed insertion target (``gears_and_base``), the held
medium gear, and a dome light for visualization. The Franka embodiment is configured with
``concatenate_observation_terms=True`` to provide a flat observation vector suitable for learned RL policies.

.. code-block:: python

   if args_cli.teleop_device is not None:
       teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
   else:
       teleop_device = None

This follows the standard Arena environment pattern: teleoperation is optional, and policy
evaluation uses ``None``.

**2. Position the Objects**

.. code-block:: python

   table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
   assembled_board.set_initial_pose(
       Pose(position_xyz=(0.88, 0.15, -0.009), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
   )
   medium_gear.set_initial_pose(Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
   gears_and_base.set_initial_pose(
       Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
   )

Before we create the scene, we need to place our objects in the right locations. The table sits at
the workspace origin, the assembled board provides visual context, and the gear base defines the
target peg pose that the policy must reach.

**3. Compose the Scene**

.. code-block:: python

    scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

Now we bring everything together into an IsaacLab-Arena scene.
See :doc:`../../concepts/scene/index` for scene composition details.

**4. Create the Gear Insertion RL Task**

.. code-block:: python

    task = NistGearInsertionRLTask(
        assembled_board=assembled_board,
        held_gear=medium_gear,
        background_scene=table,
        gear_base_asset=gears_and_base,
        geometry_cfg=geometry_cfg,
        episode_length_s=15.0,
        grasp_cfg=embodiment.get_gear_insertion_grasp_config(),
        fingertip_body_name=embodiment.get_command_body_name(),
        enable_randomization=True,
        disable_success_termination=args_cli.disable_success_termination,
    )

The ``NistGearInsertionRLTask`` encapsulates the policy objective: align the held gear with the
target peg and insert it successfully. The task includes:

- **Reward Terms**: Dense rewards for keypoint alignment between the held-gear base and the peg, plus an insertion-geometry bonus
- **Observation Space**: Task observations (peg pose, held-gear base pose, peg-to-gear delta) and the 24-D OSC policy observation
- **Termination Conditions**: Timeout and geometric insertion success
- **Success Condition**: Held gear seated on the peg within tolerance (optionally disabled with ``--disable_success_termination`` for uninterrupted rollouts)

See :doc:`../../concepts/task/index` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,
       env_cfg_callback=mdp.assembly_env_cfg_callback,
   )

Finally, we assemble all the pieces into a complete, runnable RL environment. The ``IsaacLabArenaEnvironment``
connects the embodiment (the robot), the scene (the world), and the task (the objective and rewards).
See :doc:`../../concepts/concept_overview` for environment composition details.


Validation: Run Zero-Action Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate the environment loads correctly, run a short zero-action rollout and check for errors:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 10 \
     --num_envs 1 \
     nist_assembled_gear_mesh_osc


If the environment is set up correctly, Isaac Lab will initialize, create the environment,
run the rollout, and exit without errors.
