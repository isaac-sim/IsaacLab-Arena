Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.6 policy in closed-loop
and evaluating it in the Arena G1 Loco-Manipulation Apple-to-Plate Task environment.


**Docker Container**: Base + GR00T (see :doc:`../imitation_learning/index` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/locomanip_apple_tutorial
    export MODELS_DIR=/models/isaaclab_arena/locomanip_apple_tutorial

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_4_policy_training>`.


Step 1: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first run the policy in a single environment with visualization via the GUI.

The GR00T model is configured by a config file at ``isaaclab_arena_gr00t/policy/config/g1_locomanip_apple_gr00t_closedloop_config.yaml``.

.. dropdown:: Configuration file (``g1_locomanip_apple_gr00t_closedloop_config.yaml``):
   :animate: fade-in

   .. code-block:: yaml

      model_path: /models/isaaclab_arena/locomanip_apple_tutorial/checkpoint-20000
      language_instruction: "Pick up the apple from the shelf, and place it onto the plate on the table located at the right of the shelf."
      action_horizon: 50
      embodiment_tag: NEW_EMBODIMENT
      video_backend: decord
      modality_config_path: isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py

      policy_joints_config_path: isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml

      action_chunk_length: 50
      pov_cam_name_sim: "robot_head_cam_rgb"

      task_mode_name: g1_locomanipulation

Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_apple_gr00t_closedloop_config.yaml \
     --num_steps 1500 \
     --device cpu \
     --enable_cameras \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --embodiment g1_wbc_joint

The evaluation should produce the following output on the console at the end of the evaluation.
You should see similar metrics.

Note that all these metrics are computed over the entire evaluation process, and are affected
by the quality of post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. code-block:: text

   [Rank 0/1] Metrics: {'success_rate': 1.0, 'num_episodes': 1}

Step 2: Run Parallel Environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple parallel environments is also supported by the policy runner.

.. tab-set::

   .. tab-item:: Single GPU Evaluation

      Test the policy in 5 parallel environments with visualization via the GUI run:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --viz kit \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_apple_gr00t_closedloop_config.yaml \
           --num_steps 1200 \
           --num_envs 5 \
           --enable_cameras \
           --device cuda \
           --policy_device cuda  \
           galileo_g1_locomanip_pick_and_place \
           --object apple_01_objaverse_robolab \
           --destination clay_plates_hot3d_robolab \
           --embodiment g1_wbc_joint

   .. tab-item:: Distribute Multi-GPU Evaluation

      Test the policy in 5 parallel environments on each GPU with 2 GPUs total run:

      .. code-block:: bash

         python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_apple_gr00t_closedloop_config.yaml \
           --num_steps 1200 \
           --num_envs 5 \
           --enable_cameras \
           --device cuda \
           --policy_device cuda  \
           --distributed \
           --headless \
           galileo_g1_locomanip_pick_and_place \
           --object apple_01_objaverse_robolab \
           --destination clay_plates_hot3d_robolab \
           --embodiment g1_wbc_joint


And during the evaluation, you should see the following output on the console at the end of the evaluation
indicating which environments are terminated (task-specific conditions like the apple is placed onto the plate,
or the episode length is exceeded by 30 seconds),
or truncated (if timeouts are enabled, like the maximum episode length is exceeded).

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([3], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate might not be 1.0 as more trials are being evaluated and randomizations are being introduced,
and the number of episodes is more than the single environment evaluation because of the parallelization.

.. code-block:: text

   [Rank 0/1] Metrics: {'success_rate': 1.0, 'num_episodes': 4}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``g1_wbc_joint``, which is different
   from ``g1_wbc_pink`` used in data generation.
   This is because during tele-operation, the upper body is controlled via target end-effector poses,
   which are realized by using the PINK IK controller, and the lower body is controlled via a WBC policy.
   GR00T N1.6 policy is trained on upper body joint positions and lower body WBC policy inputs, so we use
   ``g1_wbc_joint`` for closed-loop policy inference.

.. note::

   The example policy was trained on datasets generated with CPU-based physics, so the
   single-environment command above uses ``--device cpu`` to keep evaluation physics aligned
   with training and give per-episode reproducibility. The parallel commands instead use
   ``--device cuda`` for throughput -- this swaps the physics backend, so individual episodes
   are no longer bit-for-bit reproducible against the CPU-trained policy, but aggregate
   success-rate metrics over many episodes remain informative. If your dataset was generated on
   GPU physics, prefer ``--device cuda`` for both single and parallel runs to keep evaluation
   physics aligned with training.

.. note::

   The apple-to-plate task is typically harder than the brown-box-to-blue-bin task: the apple is
   smaller and rounder (easier to nudge or roll off the edge), and a ~30 cm flat plate leaves far
   less drop margin than the deep blue bin. Both variants share the same contact-sensor success
   termination (``force_threshold=0.5 N``, ``velocity_threshold=0.1 m/s``), filtered to contacts
   with the ``--destination`` asset. Both values are passed to ``LocomanipPickAndPlaceTask`` from
   ``isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py``; edit the
   ``force_threshold`` / ``velocity_threshold`` kwargs there if you need a different success
   criterion for a new apple/plate combination.
