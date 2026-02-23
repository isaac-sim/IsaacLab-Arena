Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.6 policy in closed-loop
and evaluating it in Arena GR1 Open Microwave Door Task environment.

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/sequential_static_manipulation_tutorial
    export MODELS_DIR=/models/isaaclab_arena/sequential_static_manipulation_tutorial


Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_4_policy_training>` or downloaded the
pre-trained model checkpoint below:

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   These commands can be used to download the pre-trained GR00T N1.6 policy checkpoint,
   such that the preceding steps can be skipped.

   .. code-block:: bash

      hf download \
        nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task \
        --include "ranch_bottle_into_fridge/*" \
        --repo-type model \
        --local-dir $MODELS_DIR/checkpoint-20000


Step 1: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first run the policy in a single environment with visualization via the GUI.

The GR00T model is configured by a config file at ``isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml``.

.. dropdown:: Configuration file (``gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml``):
   :animate: fade-in

   .. code-block:: yaml

      model_path: /models/isaaclab_arena/sequential_static_manipulation_tutorial/checkpoint-20000

      language_instruction: "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
      action_horizon: 16
      embodiment_tag: GR1
      video_backend: decord
      modality_config_path: isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py

      policy_joints_config_path: isaaclab_arena_gr00t/embodiments/gr1/gr00t_26dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/embodiments/gr1/36dof_joint_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/embodiments/gr1/54dof_joint_space.yaml
      action_chunk_length: 16
      task_mode_name: gr1_tabletop_manipulation

      pov_cam_name_sim: "robot_pov_cam_rgb"

      original_image_size: [512, 512, 3]
      target_image_size: [512, 512, 3]


Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
     --enable_cameras \
     put_item_in_fridge_and_close_door \
     --embodiment gr1_joint \
     --object ranching_dressing_bottle

The evaluation should produce the following output on the console at the end of the evaluation.
At the end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate for this sequential task, object moved rate for the first subtask,
and the revolute joint moved rate for the second subtask, and the subtask success rate for each subtask.
You should see similar metrics. All of them shall be greater than 0.9, and the number of episodes should be in the range of 3-6.

Note that all these metrics are computed over the entire evaluation process, and are affected by the quality of
post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. tabs::

   .. tab:: Best Quality (OSMO)

      .. code-block:: text

         Metrics: Metrics: {'success_rate': 1.0, 'object_moved_rate_subtask_0': 1.0, 'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [1.0, 1.0], 'num_episodes': 5}

.. todo::

   1. Verify evaluation using single-gpu finetune command in the last step.


Step 2: Run Parallel environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple parallel environments is also supported by the policy runner.

.. tabs::

   .. tab:: Single GPU Evaluation

      Test the policy in 10 parallel environments with visualization via the GUI run:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml \
           --num_steps 2000 \
           --num_envs 10 \
           --enable_cameras \
           put_item_in_fridge_and_close_door \
           --embodiment gr1_joint \
           --object ranching_dressing_bottle

   .. tab:: Distribute Multi-GPU Evaluation

      Test the policy in 10 parallel environments on each GPU with 2 GPUs total run:

      .. code-block:: bash

         python -m torch.distributed.run --nnode=1 --nproc_per_node=2 isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml \
           --num_steps 2000 \
           --num_envs 10 \
           --enable_cameras \
           --headless \
           --distributed \
           put_item_in_fridge_and_close_door \
           --embodiment gr1_joint \
           --object ranch_dressing_bottle


And during the evaluation, you should see the following output on the console at the end of the evaluation
indicating which environments are terminated (task-specific conditions like the microwave door is opened),
or truncated (if timeouts are enabled, like the maximum episode length is exceeded).

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([7], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate for this sequential task, object moved rate for the first subtask,
and the revolute joint moved rate for the second subtask, and the subtask success rate for each subtask.
All of them might not be 1.0 as more trials are being evaluated, and the number of episodes is more
than the single environment evaluation because of the parallel evaluation.

.. code-block:: text

   Metrics: {'success_rate': 0.98, 'object_moved_rate_subtask_0': 1.0, 'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [0.98, 1.0], 'num_episodes': 50}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``gr1_joint``, which is different
   from ``gr1_pink`` used in data generation.
   This is because during tele-operation, the robot is controlled via target end-effector poses,
   which are realized by using the PINK IK controller.
   GR00T N1.6 policy is trained on upper body joint positions, so we use
   ``gr1_joint`` for closed-loop policy inference.


Step 3: Multi-object Heterogeneous Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step demonstrates evaluation of the policy in heterogeneous environments with multiple objects.

.. tabs::

   .. tab:: Single GPU Evaluation

      Test the policy in 10 parallel environments with visualization via the GUI run:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
         --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
         --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
         --num_steps 2000 \
         --num_envs 10 \
         --enable_cameras \
         put_item_in_fridge_and_close_door \
         --embodiment gr1_joint \
         --object_set ketchup_bottle ranch_dressing_bottle bbq_sauce_bottle mayonnaise_bottle

   .. tab:: Distribute Multi-GPU Evaluation

      Test the policy in 10 parallel environments on each GPU with 2 GPUs total run:

      .. code-block:: bash

         python -m torch.distributed.run --nnode=1 --nproc_per_node=2 isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
           --num_steps 2000 \
           --num_envs 10 \
           --enable_cameras \
           --distributed \
           --headless \
           put_item_in_fridge_and_close_door \
           --embodiment gr1_joint \
           --object_set ketchup_bottle ranch_dressing_bottle bbq_sauce_bottle mayonnaise_bottle

Each environment has a different object spawned from the object set. The same policy is used for all those environments.
At then end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate for this sequential task, object moved rate for the first subtask,
and the revolute joint moved rate for the second subtask, and the subtask success rate for each subtask.
All of them might not be 1.0 as unseen objects are being evaluated and the policy is not trained on them.

.. code-block:: text

   Metrics: {'success_rate': 0.8666666666666667, 'object_moved_rate_subtask_0': 1.0, 'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [1.0, 1.0], 'num_episodes': 30}

Step 4: Batch Evaluation with JSON Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more systematic evaluation, you can use the ``eval_runner.py`` script to evaluate the policy in a batch.

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py --config eval_config.json

This will automatically evaluate the policy in the batch and output the metrics.


**1. Create Evaluation Configuration**

Create a file ``eval_jobs_config.json``:

.. code-block:: json

   {
     "jobs": [
       {
         "name": "gr1_put_ranch_dressing_bottle_in_fridge_and_close_door",
         "arena_env_args": {
          "enable_cameras": true,
           "environment": "put_item_in_fridge_and_close_door",
           "object": "ranch_dressing_bottle",
           "embodiment": "gr1_joint"
         },
         "num_envs": 10,
         "num_steps": 500,
         "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
         "policy_config_dict": {
           "policy_config_yaml_path": "isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml",
           "policy_device": "cuda:0"
         }
       },
       {
         "name": "gr1_put_mayonnaise_bottle_in_fridge_and_close_door",
         "arena_env_args": {
           "enable_cameras": true,
           "environment": "put_item_in_fridge_and_close_door",
           "object": "mayonnaise_bottle",
           "embodiment": "gr1_joint"
         },
         "num_envs": 10,
         "num_steps": 500,
         "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
         "policy_config_dict": {
           "policy_config_yaml_path": "isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml",
           "policy_device": "cuda:0"
         }
       }
     ]
   }

Run the batch evaluation:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py --config eval_jobs_config.json

This will automatically evaluate the policy with the given configuration and output the metrics.

.. code-block:: text

   +--------------------------------------------------------+-----------+---------------------------------------------------------------------------+----------+-----------+
   |                        Job Name                        |   Status  |                                Policy Type                                | Num Envs | Num Steps |
   +--------------------------------------------------------+-----------+---------------------------------------------------------------------------+----------+-----------+
   | gr1_put_ranch_dressing_bottle_in_fridge_and_close_door | completed | isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy |    1     |    500    |
   |   gr1_put_mayonnaise_bottle_in_fridge_and_close_door   | completed | isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy |    1     |    500    |
   +--------------------------------------------------------+-----------+---------------------------------------------------------------------------+----------+-----------+

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   gr1_put_mayonnaise_bottle_in_fridge_and_close_door:
   num_episodes                            1
   object_moved_rate_subtask_0        1.0000
   revolute_joint_moved_rate_subtask_1     1.0000
   subtask_success_rate           [1.0, 1.0]
   success_rate                       1.0000

   gr1_put_ranch_dressing_bottle_in_fridge_and_close_door:
   num_episodes                            1
   object_moved_rate_subtask_0        1.0000
   revolute_joint_moved_rate_subtask_1     1.0000
   subtask_success_rate           [1.0, 1.0]
   success_rate                       1.0000
   ======================================================================
