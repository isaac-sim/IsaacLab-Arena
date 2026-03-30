Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.6 policy in closed-loop
and evaluating it in Arena GR1 Open Microwave Door Task environment.

**Docker Container**: Base + GR00T (see :doc:`../imitation_learning/index` for more details)

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

      mkdir -p $MODELS_DIR/checkpoint-20000
      hf download \
        nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task \
        --include "ranch_bottle_into_fridge/*" \
        --repo-type model \
        --local-dir $MODELS_DIR/_hf_download
      mv $MODELS_DIR/_hf_download/ranch_bottle_into_fridge/* $MODELS_DIR/checkpoint-20000/
      rm -rf $MODELS_DIR/_hf_download


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
     --object ranch_dressing_hope_robolab

The evaluation should produce the following output on the console at the end of the evaluation.
At the end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate for this sequential task, object moved rate for the first subtask,
and the revolute joint moved rate for the second subtask, and the subtask success rate for each subtask.
You should see similar metrics. All of them shall be greater than 0.9, and the number of episodes should be in the range of 3-6.

Note that all these metrics are computed over the entire evaluation process, and are affected by the quality of
post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. tabs::

   .. tab:: Best Quality

      .. code-block:: text

         Metrics: Metrics: {'success_rate': 1.0, 'object_moved_rate_subtask_0': 1.0, 'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [1.0, 1.0], 'num_episodes': 5}

   .. tab:: Low Hardware Requirements

      Evaluated with checkpoint-30000, instead of checkpoint-20000 referenced in the policy configuration file.

      .. code-block:: text

         Metrics: Metrics: {'success_rate': 0.75, 'object_moved_rate_subtask_0': 1.0, 'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [0.75, 0.75], 'num_episodes': 4}

Step 2: Run Parallel environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple parallel environments is also supported by the policy runner.

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
           --object ranch_dressing_hope_robolab

   .. tab:: Distribute Multi-GPU Evaluation

      Test the policy in 10 parallel environments on each GPU with 2 GPUs total run:

      .. code-block:: bash

         python -m torch.distributed.run --nnode=1 --nproc_per_node=2 isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
           --num_steps 2000 \
           --num_envs 10 \
           --enable_cameras \
           --headless \
           --distributed \
           put_item_in_fridge_and_close_door \
           --embodiment gr1_joint \
           --object ranch_dressing_hope_robolab


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
         --object_set ketchup_bottle_hope_robolab ranch_dressing_hope_robolab bbq_sauce_bottle_hope_robolab mayonnaise_bottle_hope_robolab

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
           --object_set ketchup_bottle_hope_robolab ranch_dressing_hope_robolab bbq_sauce_bottle_hope_robolab mayonnaise_bottle_hope_robolab

Each environment has a different object spawned from the object set. The same policy is used for all those environments.
At then end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate for this sequential task, object moved rate for the first subtask,
and the revolute joint moved rate for the second subtask, and the subtask success rate for each subtask.

.. code-block:: text

   Metrics: {'success_rate': 0.8666666666666667, 'object_moved_rate_subtask_0': 1.0, 'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [1.0, 1.0], 'num_episodes': 30}

The policy demonstrates robust intra-class generalization, achieving an average success rate of 86.67%
when evaluated on structurally similar sauce bottles.
While this represents a slight performance regression compared to the training object,
it indicates that the model has successfully learned geometric features common to the category rather than
just over-fitting to a single object mesh.


Step 4: Sequential Batch Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sequential batch of jobs, i.e. different tasks, objects, embodiments or policies, can be evaluated by the ``eval_runner.py`` script.
It minimizes the overhead of reloading system modules and environment classes for each job while keeping the simulation application alive.
The evaluation batch can be specified in a config file, with examples shown below.

.. dropdown:: Configuration file (``gr1_sequential_static_manip_eval_jobs_config.json``):
   :animate: fade-in

   .. code-block:: json

      {
      "jobs": [
         {
            "name": "gr1_put_ranch_dressing_bottle_in_fridge_and_close_door",
            "arena_env_args": {
               "num_envs": 10,
               "enable_cameras": true,
               "environment": "put_item_in_fridge_and_close_door",
               "object": "ranch_dressing_hope_robolab",
               "embodiment": "gr1_joint"
            },
            "num_steps": 500,
            "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
            "policy_config_dict": {
            "policy_config_yaml_path": "isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml",
            "policy_device": "cuda:0"
            }
         },
         {
            "name": "gr1_put_jug_in_fridge_and_close_door",
            "arena_env_args": {
               "num_envs": 10,
               "enable_cameras": true,
               "environment": "put_item_in_fridge_and_close_door",
               "object": "jug",
               "embodiment": "gr1_joint"
            },
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

   python isaaclab_arena/evaluation/eval_runner.py --eval_jobs_config isaaclab_arena_gr00t/policy/config/gr1_sequential_static_manip_eval_jobs_config.json

This will automatically evaluate the policy with the given configuration and output the metrics.
You should see the following output on the console indicating the jobs and metrics.

.. code-block:: text

   +---------------------------------------------------------+------------+----------------------------------------------------------------------------+----------+-----------+--------------+
   | Job Name                                                | Status     | Policy Type                                                                | Num Envs | Num Steps | Num Episodes |
   +---------------------------------------------------------+------------+----------------------------------------------------------------------------+----------+-----------+--------------+
   || gr1_put_jug_in_fridge_and_close_door                   || completed || isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy || 10      || 500      || None        |
   || gr1_put_ranch_dressing_bottle_in_fridge_and_close_door || completed || isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy || 10      || 500      || None        |
   +---------------------------------------------------------+------------+----------------------------------------------------------------------------+----------+-----------+--------------+

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   gr1_put_jug_in_fridge_and_close_door:
   num_episodes                           10
   object_moved_rate_subtask_0        1.0000
   revolute_joint_moved_rate_subtask_1     1.0000
   subtask_success_rate           [0.2, 0.2]
   success_rate                       0.1000

   gr1_put_ranch_dressing_bottle_in_fridge_and_close_door:
   num_episodes                           10
   object_moved_rate_subtask_0        1.0000
   revolute_joint_moved_rate_subtask_1     1.0000
   subtask_success_rate           [0.9, 0.9]
   success_rate                       0.9000
   ======================================================================

With the policy trained on using ranch dressing bottle as object of interest,
the success rate for generalizing to putting the unseen object, jug, in the fridge is 0.0.
This is expected as the policy is not trained on the jug, comparing to the success rate of 0.8 for the trained object, ranch dressing bottle.

Step 5: Remote Policy Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same task can also be evaluated using a remote policy running in a
separate process, using the generic remote policy interface described
in :doc:`../../concepts/concept_remote_policies_design`.

Start the remote policy server (for example, a GR00T-based policy) in a
separate terminal using the provided helper script:

.. code-block:: bash

   bash docker/run_gr00t_server.sh \
     # include model directory mount via -m flag, e.g. -m /models/
     --host 127.0.0.1 \
     --port 5555 \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy \
     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml

Then, instead of running the closed-loop policy directly inside the
Arena process, connect from the evaluation script using a client-side
remote policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy \
     --remote_host 127.0.0.1 \
     --remote_port 5555 \
     --num_steps 2000 \
     --num_envs 10 \
     --enable_cameras \
     --remote_kill_on_exit \
     put_item_in_fridge_and_close_door \
     --embodiment gr1_joint \
     # If you want to evaluate on a single object, you can specify it here
     --object ranch_dressing_hope_robolab
     # If you want to evaluate among multiple objects following heterogeneous evaluation, you can specify it here
     # --object_set ketchup_bottle_hope_robolab ranch_dressing_hope_robolab bbq_sauce_bottle_hope_robolab mayonnaise_bottle_hope_robolab

With this setup, the environment and evaluation run in the base container,
and GR00T inference runs in a separate server process, connected via the
remote policy interface. The client-side policy depends only on Isaac Lab
Arena; the base container does not need GR00T or its Python dependencies.
