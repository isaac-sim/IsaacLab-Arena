Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.5 policy in closed-loop
and evaluating it across multiple parallel environments.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_4_policy_training>` or downloaded the
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
         nvidia/GN1x-Tuned-Arena-GR1-Manipulation \
         --local-dir <CHECKPOINTS_DIR>



Step 1: Configure Closed-Loop Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create or verify the inference configuration file:

**Configuration** (``isaaclab_arena/arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml``):

.. code-block:: yaml

   # Model configuration
   model_path: /checkpoints/GN1x-Tuned-Arena-GR1-Manipulation
   embodiment_tag: gr1
   data_config: gr1_arms_only

   # Task configuration
   language_instruction: "Reach out to the microwave and open it."
   task_mode_name: gr1_manipulation

   # Inference parameters
   denoising_steps: 10
   policy_device: cuda
   target_image_size: [256, 256, 3]

   # Joint mappings
   gr00t_joints_config_path: isaaclab_arena/policy/config/gr1/gr00t_26dof_joint_space.yaml
   action_joints_config_path: isaaclab_arena/policy/config/gr1/36dof_joint_space.yaml
   state_joints_config_path: isaaclab_arena/policy/config/gr1/54dof_joint_space.yaml


.. todo:: (alexmillane, 2025-10-23): See if we can move the model path out of the config file
   and onto the command line. Then change the statement above.



Step 2: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena/arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml \
     --num_steps 400 \
     --enable_cameras \
     gr1_open_microwave \
     --embodiment gr1_joint

The evaluation should produce the following output on the console at the end of the evaluation.
You should see similar metrics.

.. code-block:: text

   Metrics: {success_rate: 1.0, door_moved_rate: 1.0, num_episodes: 2}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``gr1_joint``, which is different
   from ``gr1_pink`` used in data generation.
   This is because during tele-operation, the robot is controlled via target end-effector poses,
   which are realized by using the PINK IK controller.
   GR00T N1.5 policy is trained on upper body joint positions, so we use
   ``gr1_joint`` for closed-loop policy inference.



Step 3: Run Parallel Evaluation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IsaacLab Arena supports evaluating the policy in parallel across multiple environments.
To perform a parallel evaluation across 16 environments, we set ``num_envs`` to 16,
by running the following command.

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena/arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml \
     --num_steps 400 \
     --num_envs 16 \
     --enable_cameras \
     --headless \
     gr1_open_microwave \
     --embodiment gr1_joint

The evaluation should produce the following output on the console at the end of the evaluation.
You should see similar metrics.

.. code-block:: text

   Metrics: {success_rate: 0.9375, door_moved_rate: 0.9375, num_episodes: 32}
