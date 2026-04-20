Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

On this page we briefly describe the RL environment used in this example workflow
and validate that we can load it in Isaac Lab.

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: The NIST Gear Insertion Environment (simplified)
   :animate: fade-in

   The snippet below is a simplified view of the environment definition. For the full
   implementation — including controller configuration, gripper action, grasp config,
   and additional scene setup — see ``nist_assembled_gearmesh_osc_environment.py``.

   .. code-block:: python

      class NISTAssembledGearMeshOSCEnvironment(ExampleEnvironmentBase):

          name: str = "nist_assembled_gear_mesh_osc"

          def get_env(self, args_cli: argparse.Namespace):
              table = self.asset_registry.get_asset_by_name("table")()
              assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
              gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
              medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
              light = self.asset_registry.get_asset_by_name("light")()

              embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                  enable_cameras=args_cli.enable_cameras,
                  concatenate_observation_terms=True,
              )

              embodiment.action_config.arm_action = NistGearInsertionOscActionCfg(
                  asset_name="robot",
                  joint_names=["panda_joint[1-7]"],
                  body_name="panda_fingertip_centered",
                  fixed_asset_name=gears_and_base.name,
                  peg_offset=(0.02025, 0.0, 0.025),
              )

              embodiment.observation_config.policy.nist_gear_policy_obs = ObsTerm(
                  func=NistGearInsertionPolicyObservations,
                  params={
                      "robot_name": "robot",
                      "board_name": gears_and_base.name,
                      "peg_offset": [0.02025, 0.0, 0.025],
                  },
              )

              task = NistGearInsertionTask(
                  assembled_board=assembled_board,
                  held_gear=medium_gear,
                  background_scene=table,
                  peg_offset_from_board=[0.02025, 0.0, 0.0],
                  peg_offset_for_obs=[0.02025, 0.0, 0.025],
                  gear_base_asset=gears_and_base,
                  episode_length_s=15.0,
                  enable_randomization=True,
                  rl_training_mode=args_cli.rl_training_mode,
              )

              table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
              assembled_board.set_initial_pose(
                  Pose(position_xyz=(0.88, 0.15, -0.009), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
              )
              medium_gear.set_initial_pose(Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
              gears_and_base.set_initial_pose(
                  Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
              )

              scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

              return IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=task,
                  rl_framework=RLFramework.RL_GAMES,
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

Here, we're selecting the components needed for our RL task: a table as the support surface,
the assembled NIST board for context, the fixed insertion target (``gears_and_base``), and the
held medium gear that the robot must insert onto the peg. The Franka embodiment is configured
with ``concatenate_observation_terms=True`` so the observation groups can be consumed by the RL stack.

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

Before we create the scene, we place the assets in the assembled-board layout used for this task.
The table provides the workspace, the assembled board provides visual context, and the gear base
defines the target peg pose that the policy must reach relative to.

**3. Configure the Franka Embodiment for Operational-Space Control**

.. code-block:: python

   embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
       enable_cameras=args_cli.enable_cameras,
       concatenate_observation_terms=True,
   )

   embodiment.action_config.arm_action = NistGearInsertionOscActionCfg(
       asset_name="robot",
       joint_names=["panda_joint[1-7]"],
       body_name="panda_fingertip_centered",
       fixed_asset_name=gears_and_base.name,
       peg_offset=(0.02025, 0.0, 0.025),
   )

The arm is controlled in operational space rather than joint position space.
The policy emits a 7-D action:

- 3 position commands
- 3 rotation commands
- 1 auxiliary success-prediction scalar used by the task-specific controller/reward stack

The action term smooths commands, clips task-space deltas, locks roll and pitch to the
assembly convention, and defines targets relative to the peg position. This makes the
action space better aligned with insertion than a generic joint-space controller.

**4. Configure the Policy Observation**

.. code-block:: python

   embodiment.observation_config.policy.nist_gear_policy_obs = ObsTerm(
       func=NistGearInsertionPolicyObservations,
       params={
           "robot_name": "robot",
           "board_name": gears_and_base.name,
           "peg_offset": [0.02025, 0.0, 0.025],
       },
   )

The environment swaps out the default embodiment policy observations for a specialized 24-D
observation stack designed for insertion. It includes:

- fingertip pose relative to the fixed asset
- end-effector linear and angular velocity
- wrist-force feedback
- a sampled force threshold
- previous actions

Task observations are still provided separately for critic/state use.

**5. Create the Gear Insertion Task**

.. code-block:: python

   task = NistGearInsertionTask(
       assembled_board=assembled_board,
       held_gear=medium_gear,
       background_scene=table,
       peg_offset_from_board=[0.02025, 0.0, 0.0],
       peg_offset_for_obs=[0.02025, 0.0, 0.025],
       gear_base_asset=gears_and_base,
       success_z_fraction=0.20,
       xy_threshold=0.0025,
       episode_length_s=15.0,
       enable_randomization=True,
       rl_training_mode=args_cli.rl_training_mode,
   )

The ``NistGearInsertionTask`` encapsulates the RL training objective: align the held gear with the
target peg and insert it successfully. The task includes:

- **Reward Terms**: Dense shaping for alignment, engagement, insertion success, and action/contact regularization
- **Observation Space**: Task-specific policy observations, plus task observations for critic/state
- **Termination Conditions**: Timeout and insertion success, with success disabled during training by ``--rl_training_mode``
- **Success Metric**: ``success_rate`` computed during evaluation

When ``enable_randomization=True``, the task also configures environment-side randomization through
reset/startup events, including fixed-asset yaw variation, friction/material changes for the gear,
robot, and fixed asset, and held-object mass perturbations.

See :doc:`../../concepts/concept_tasks_design` for task creation details.

**6. Compose the Scene and Create the IsaacLab Arena Environment**

.. code-block:: python

   scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

   return IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=task,
       rl_framework=RLFramework.RL_GAMES,
       rl_policy_cfg="isaaclab_arena_examples.policy:nist_gear_insertion_osc_rl_games.yaml",
   )

Finally, we assemble all the pieces into a complete, runnable RL environment. The
``IsaacLabArenaEnvironment`` connects the embodiment (the robot), the scene (the world),
and the task (the objective, rewards, and metrics), and declares that this workflow uses
the RL Games training stack.
See :doc:`../../concepts/concept_environment_design` for environment composition details.


Validation: Run One Training Iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate that the environment loads correctly, run one training iteration and check for errors:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train_rl_games.py \
     --headless \
     --num_envs 128 \
     --max_iterations 1 \
     --agent_cfg_path isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml \
     nist_assembled_gear_mesh_osc \
     --rl_training_mode

If the environment is set up correctly, RL Games should initialize, create the environments,
run one optimization iteration, and then exit without environment-construction errors.

You should see output indicating that training has started and that one iteration completed.

.. image:: ../../../images/nist_gear_insertion_task.gif
   :align: center
   :height: 400px
