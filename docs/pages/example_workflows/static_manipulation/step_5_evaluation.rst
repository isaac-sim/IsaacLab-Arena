Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.5 policy in closed-loop
and evaluating it in Arena GR1 Open Microwave Door Task environment.

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
      action_chunk_length: 16
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
You should see similar metrics. The success rate shall be more than 0.8, and the door moved rate
should be greater than 0.9, and the number of episodes should be in the range of 10-20.

Note that all these metrics are computed over the entire evaluation process, and are affected by the quality of
post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. tabs::

   .. tab:: Best Quality

      .. code-block:: text

         Metrics: {'success_rate': 0.8823529411764706, 'revolute_joint_moved_rate': 1.0, 'num_episodes': 17}

   .. tab:: Low Hardware Requirements

      .. code-block:: text

         Metrics: {'success_rate': 1.0, 'revolute_joint_moved_rate': 1.0, 'num_episodes': 19}


Step 2: Run Parallel environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple parallel environments is also supported by the policy runner.

Test the policy in 10 parallel environments with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
     --num_envs 10 \
     --enable_cameras \
     gr1_open_microwave \
     --embodiment gr1_joint

And during the evaluation, you should see the following output on the console at the end of the evaluation
indicating which environments are terminated (task-specific conditions like the microwave door is opened),
or truncated (if timeouts are enabled, like the maximum episode length is exceeded).

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([7], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate and door moved rate might not be 1.0 as more trials are being evaluated, and the number of episodes is more
than the single environment evaluation because of the parallel evaluation.

.. code-block:: text

   Metrics: {'success_rate': 0.605, 'revolute_joint_moved_rate': 0.955, 'num_episodes': 200}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``gr1_joint``, which is different
   from ``gr1_pink`` used in data generation.
   This is because during tele-operation, the robot is controlled via target end-effector poses,
   which are realized by using the PINK IK controller.
   GR00T N1.5 policy is trained on upper body joint positions, so we use
   ``gr1_joint`` for closed-loop policy inference.
