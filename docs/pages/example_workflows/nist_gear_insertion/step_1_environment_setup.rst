Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

On this page we briefly describe the RL environment used in this example workflow
and validate that we can load it in Isaac Lab.

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The ``nist_assembled_gear_mesh_osc`` environment builds a Franka Panda
assembly scene where the robot starts with the medium gear in its gripper and
learns to align it with the target peg on the NIST board.

The environment is defined in
``isaaclab_arena_environments/nist_assembled_gearmesh_osc_environment.py``:

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
              from isaaclab_arena.tasks.nist_gear_insertion_task import (
                  GearInsertionGeometryCfg,
                  GraspCfg,
                  NistGearInsertionTask,
              )
              from isaaclab_arena.utils.pose import Pose

              peg_tip_offset = (0.02025, 0.0, 0.025)
              peg_base_offset = (0.02025, 0.0, 0.0)
              success_z_fraction = 0.20
              xy_threshold = 0.0025
              episode_length_s = 15.0

              table = self.asset_registry.get_asset_by_name("table")()
              assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
              gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
              medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
              light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
              light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

              embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                  enable_cameras=args_cli.enable_cameras,
                  concatenate_observation_terms=True,
                  fixed_asset_name=gears_and_base.name,
                  peg_offset=peg_tip_offset,
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

              grasp_cfg = GraspCfg(**embodiment.get_gear_insertion_grasp_config())
              geometry_cfg = GearInsertionGeometryCfg(
                  peg_offset_from_board=list(peg_base_offset),
                  peg_offset_for_obs=list(peg_tip_offset),
                  success_z_fraction=success_z_fraction,
                  xy_threshold=xy_threshold,
              )

              task = NistGearInsertionTask(
                  assembled_board=assembled_board,
                  held_gear=medium_gear,
                  background_scene=table,
                  gear_base_asset=gears_and_base,
                  geometry_cfg=geometry_cfg,
                  episode_length_s=episode_length_s,
                  grasp_cfg=grasp_cfg,
                  enable_randomization=True,
                  rl_training_mode=args_cli.rl_training_mode,
              )

              return IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=task,
                  teleop_device=teleop_device,
                  env_cfg_callback=mdp.assembly_env_cfg_callback,
                  rl_framework_entry_point="rl_games_cfg_entry_point",
                  rl_policy_cfg="isaaclab_arena_examples.policy:nist_gear_insertion_osc_rl_games.yaml",
              )


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset Registry**

.. code-block:: python

   table = self.asset_registry.get_asset_by_name("table")()
   assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
   gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
   medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
   light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
   light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

Here, we're selecting the components needed for the RL task: a table as the
support surface, the assembled NIST board for context, the fixed insertion
target (``gears_and_base``), and the held medium gear that the robot inserts
onto the peg. The dome light uses an explicit ``DomeLightCfg`` so visualization
is consistent across runs.

**2. Configure the Franka Embodiment**

.. code-block:: python

   peg_tip_offset = (0.02025, 0.0, 0.025)

   embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
       enable_cameras=args_cli.enable_cameras,
       concatenate_observation_terms=True,
       fixed_asset_name=gears_and_base.name,
       peg_offset=peg_tip_offset,
   )

The ``franka_nist_gear_osc`` embodiment configures the robot for
operational-space control. The policy emits a 7-D action:

- 3 position commands
- 3 rotation commands
- 1 auxiliary success-prediction scalar used by the task-specific reward stack

The NIST Franka embodiment owns the task-specific OSC action term, fingertip frame,
initial joint pose, and grasp reset parameters. The action term smooths commands,
clips task-space deltas, locks roll and pitch to the assembly convention, and defines
targets relative to the peg position. This makes the action space better aligned with
insertion than a generic joint-space controller.

The embodiment also configures a specialized 24-D policy observation stack for
insertion. It includes:

- fingertip pose relative to the fixed asset
- end-effector linear and angular velocity
- wrist-force feedback
- a sampled force threshold
- previous actions

Task observations are still provided separately for critic/state use.

**3. Position the Objects and Compose the Scene**

.. code-block:: python

   table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
   assembled_board.set_initial_pose(
       Pose(position_xyz=(0.88, 0.15, -0.009), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
   )
   medium_gear.set_initial_pose(Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
   gears_and_base.set_initial_pose(
       Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
   )

   scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

Before we create the task, we place the assets in the assembled-board layout
used for this example. The table provides the workspace, the assembled board
provides visual context, and the gear base defines the target peg pose that the
policy must reach relative to.

See :doc:`../../concepts/scene/index` for scene composition details.

**4. Create the Gear Insertion Task**

.. code-block:: python

   peg_tip_offset = (0.02025, 0.0, 0.025)
   peg_base_offset = (0.02025, 0.0, 0.0)
   success_z_fraction = 0.20
   xy_threshold = 0.0025
   episode_length_s = 15.0

   grasp_cfg = GraspCfg(**embodiment.get_gear_insertion_grasp_config())
   geometry_cfg = GearInsertionGeometryCfg(
       peg_offset_from_board=list(peg_base_offset),
       peg_offset_for_obs=list(peg_tip_offset),
       success_z_fraction=success_z_fraction,
       xy_threshold=xy_threshold,
   )

   task = NistGearInsertionTask(
       assembled_board=assembled_board,
       held_gear=medium_gear,
       background_scene=table,
       gear_base_asset=gears_and_base,
       geometry_cfg=geometry_cfg,
       episode_length_s=episode_length_s,
       grasp_cfg=grasp_cfg,
       enable_randomization=True,
       rl_training_mode=args_cli.rl_training_mode,
   )

The ``NistGearInsertionTask`` encapsulates the RL training objective: align the held gear with the
target peg and insert it successfully. The task includes:

- **Reward Terms**: Dense shaping for alignment, engagement, insertion success, and action/contact regularization
- **Observation Space**: Task-specific policy observations, plus task observations for critic/state
- **Termination Conditions**: Timeout and insertion success, with success disabled during training by
  ``--rl_training_mode``
- **Success Metric**: ``success_rate`` computed during evaluation

When ``enable_randomization=True``, the task also configures environment-side randomization through
reset events, including fixed-asset yaw variation, robot actuator-gain variation, robot joint-friction
variation, and held-object mass perturbations.

See :doc:`../../concepts/task/index` for task creation details.

**5. Create the IsaacLab Arena Environment**

.. code-block:: python

   return IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,
       env_cfg_callback=mdp.assembly_env_cfg_callback,
       rl_framework_entry_point="rl_games_cfg_entry_point",
       rl_policy_cfg="isaaclab_arena_examples.policy:nist_gear_insertion_osc_rl_games.yaml",
   )

Finally, we assemble all the pieces into a complete, runnable RL environment. The
``IsaacLabArenaEnvironment`` connects the embodiment (the robot), the scene (the world),
and the task (the objective, rewards, and metrics), and declares that this workflow uses
the RL Games training stack.
See :doc:`../../concepts/concept_environment_compilation` for environment composition details.


Validation: Run One Training Iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate that the environment loads correctly, run one training iteration and check for errors:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train_rl_games.py \
     --task nist_assembled_gear_mesh_osc \
     --num_envs 128 \
     --max_iterations 1 \
     --rl_training_mode

If the environment is set up correctly, RL Games should initialize, create the environments,
run one optimization iteration, and then exit without environment-construction errors.

You should see output indicating that training has started and that one iteration completed.

.. image:: ../../../images/nist_gear_insertion_task.gif
   :align: center
   :height: 400px
