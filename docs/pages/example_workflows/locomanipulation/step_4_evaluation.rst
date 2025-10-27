Closed-Loop Policy Inference and Evaluation
-------------------------------------------

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

This workflow demonstrates running the trained GR00T N1.5 policy in closed-loop
and evaluating it across multiple parallel environments.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_3_policy_training>` or downloaded the
pre-trained model checkpoint below:

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   These commands can be used to download the pre-trained GR00T N1.5 policy checkpoint,
   such that the preceding steps can be skipped.
   This step requires the Hugging Face CLI, which can be installed by following the
   `official instructions <https://huggingface.co/docs/huggingface_hub/installation>`_.

   To download run (replacing ``<CHECKPOINTS_DIR>`` with the actual path):

   .. code-block:: bash

      huggingface-cli download \
         nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
         --local-dir <CHECKPOINTS_DIR>


Step 1: Configure Closed-Loop Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create or verify the inference configuration file:

**Configuration** (``isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml``):

.. code-block:: yaml

   # Model configuration
   model_path: /checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation
   embodiment_tag: new_embodiment
   data_config: unitree_g1_sim_wbc

   # Task configuration
   language_instruction: "Pick up the brown box and place it in the blue bin"
   task_mode_name: g1_locomanipulation

   # Inference parameters
   denoising_steps: 10
   policy_device: cuda
   target_image_size: [256, 256, 3]

   # Joint mappings
   gr00t_joints_config_path: isaaclab_arena/policy/config/g1/gr00t_43dof_joint_space.yaml
   action_joints_config_path: isaaclab_arena/policy/config/g1/43dof_joint_space.yaml
   state_joints_config_path: isaaclab_arena/policy/config/g1/43dof_joint_space.yaml

.. todo:: (alexmillane, 2025-10-23): See if we can move the model path out of the config file
   and onto the command line. Then change the statement above.


Step 2: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
     --num_steps 1200 \
     --enable_cameras \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

The evaluation should produce the following output on the console at the end of the evaluation.
You should see similar metrics.

.. code-block:: text

   Metrics: {success_rate: 1.0, num_episodes: 1}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``g1_wbc_joint``, which is different
   from ``g1_wbc_pink`` used in data generation.
   This is because during tele-operation, the robot is controlled target via end-effector poses,
   which are realized by using the PINK IK controller.
   GR00T N1.5 policy is trained on upper body joint positions, so we use
   ``g1_wbc_joint`` for closed-loop policy inference.


Step 3: Run Parallel Evaluation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IsaacLab Arena supports evaluating the policy in parallel across multiple environments.
To perform a parallel evaluation across 16 environments, we set ``num_envs`` to 16,
by running the following command.

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
     --num_steps 1200 \
     --num_envs 16 \
     --enable_cameras \
     --headless \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

The evaluation should produce the following output on the console at the end of the evaluation.
You should see similar metrics.

.. code-block:: text

   Metrics: {success_rate: 0.75, num_episodes: 16}
