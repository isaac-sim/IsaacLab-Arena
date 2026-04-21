Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.5 policy in closed-loop
and evaluating it in Arena G1 Loco Manipulation Task environment.


**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/locomanipulation_tutorial
    export MODELS_DIR=/models/isaaclab_arena/locomanipulation_tutorial

.. note::
    The GR00T N1.5 codebase does not support running on Blackwell architecture by default. There are
    instructions `here <https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#faq>`_ to building certain packages from source to support running on these architectures.
    We have not tested these instructions, and therefore we do not recommend using
    the **Base + GR00T** container for policy post-training and evaluation on
    Blackwell architecture, like RTX 50 series, RTX Pro 6000 or DGX Spark.


Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_3_policy_training>` or downloaded the
pre-trained model checkpoint below:

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   These commands can be used to download the pre-trained GR00T N1.5 policy checkpoint,
   such that the preceding steps can be skipped.
   This step requires the Hugging Face CLI, which can be installed by following the
   `official instructions <https://huggingface.co/docs/huggingface_hub/installation>`_.

   To download run:

   .. code-block:: bash

      hf download \
         nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
         --local-dir $MODELS_DIR/checkpoint-20000


Step 1: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first run the policy in a single environment with visualization via the GUI.

The GR00T model is configured by a config file at ``isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml``.

.. dropdown:: Configuration file (``g1_locomanip_gr00t_closedloop_config.yaml``):
   :animate: fade-in

   .. code-block:: yaml

      model_path: /models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000
      language_instruction: "Pick up the brown box from the shelf, and place it into the blue bin on the table located at the right of the shelf."
      action_horizon: 16
      embodiment_tag: new_embodiment
      video_backend: decord
      data_config: unitree_g1_sim_wbc

      policy_joints_config_path: isaaclab_arena_gr00t/config/g1/gr00t_43dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/config/g1/43dof_joint_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/config/g1/43dof_joint_space.yaml

      action_chunk_length: 16
      pov_cam_name_sim: "robot_head_cam_rgb"

      task_mode_name: g1_locomanipulation

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

Note that all these metrics are computed over the entire evaluation process, and are affected
by the quality of post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. code-block:: text

   Metrics: {success_rate: 1.0, num_episodes: 1}

Step 2: Run Parallel Environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple parallel environments is also supported by the policy runner.

Test the policy in 5 parallel environments with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
     --num_steps 1200 \
     --num_envs 5 \
     --enable_cameras \
     --device cpu \
     --policy_device cuda  \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

And during the evaluation, you should see the following output on the console at the end of the evaluation
indicating which environments are terminated (task-specific conditions like the brown box is placed into the blue bin,
or the episode length is exceeded by 30 seconds),
or truncated (if timeouts are enabled, like the maximum episode length is exceeded).

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([3], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate might not be 1.0 as more trials are being evaluated and randomizations are being introduced,
and the number of episodes is more than the single environment evaluation because of the parallelization.

.. code-block:: text

   Metrics: {'success_rate': 1.0, 'num_episodes': 4}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``g1_wbc_joint``, which is different
   from ``g1_wbc_pink`` used in data generation.
   This is because during tele-operation, the upper body is controlled via target end-effector poses,
   which are realized by using the PINK IK controller, and the lower body is controlled via a WBC policy.
   GR00T N1.5 policy is trained on upper body joint positions and lower body WBC policy inputs, so we use
   ``g1_wbc_joint`` for closed-loop policy inference.

.. note::

   The policy was trained on datasets generated using CPU-based physics, therefore the evaluation uses ``--device cpu`` to ensure physics reproducibility.
   If you have GPU-generated datasets, you can switch to using GPU-based physics for evaluation by providing the ``--device cuda`` flag.
