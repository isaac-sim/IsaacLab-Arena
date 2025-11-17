Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.5 policy in closed-loop
and evaluating it in Arena GR1 Open Microwave Door Task environment.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_4_policy_training>` or downloaded the
pre-trained model checkpoint below:

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   These commands can be used to download the pre-trained GR00T N1.5 policy checkpoint,
   such that the preceding steps can be skipped.

   .. code-block:: bash

      hf download \
         nvidia/GN1x-Tuned-Arena-GR1-Manipulation \
         --local-dir $MODELS_DIR/checkpoint-20000


**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/static_manipulation_tutorial
    export MODELS_DIR=/models/isaaclab_arena/static_manipulation_tutorial

.. note::
    The GR00T N1.5 codebase does not support running on Blackwell architecture by default. There are
    instructions `here <https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#faq>`_ to building certain packages from source to support running on these architectures.
    We have not tested these instructions, and therefore we do not recommend using
    the **Base + GR00T** container for policy post-training and evaluation on
    Blackwell architecture, like RTX 50 series, RTX Pro 6000 or DGX Spark.


Step 1: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first run the policy in a single environment with visualization via the GUI.

The GR00T model is configured by a config file at ``isaaclab_arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml``.

.. dropdown:: Configuration file (``gr1_manip_gr00t_closedloop_config.yaml``):
   :animate: fade-in

   .. code-block:: yaml

      model_path: /models/isaaclab_arena/static_manipulation_tutorial/checkpoint-20000

      language_instruction: "Reach out to the microwave and open it."
      action_horizon: 16
      embodiment_tag: gr1
      video_backend: decord
      data_config: fourier_gr1_arms_only

      policy_joints_config_path: isaaclab_arena_gr00t/config/gr1/gr00t_26dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/config/gr1/36dof_joint_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/config/gr1/54dof_joint_space.yaml
      num_feedback_actions: 16
      task_mode_name: gr1_tabletop_manipulation

      pov_cam_name_sim: "robot_pov_cam_rgb"

      original_image_size: [512, 512, 3]
      target_image_size: [512, 512, 3]


Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
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
