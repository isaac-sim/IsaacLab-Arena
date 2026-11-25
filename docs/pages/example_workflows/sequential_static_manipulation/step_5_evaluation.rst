Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.6 policy in closed-loop
and evaluating it in the GR1 Sequential Pick & Place and Close Door Environment.

:runtime_setup_tabs:`--group gr00t-client`

Once the runtime is ready, set the dataset and models directories.

.. code-block:: bash

   export DATASET_DIR="$HOME/datasets/isaaclab_arena/sequential_static_manipulation_tutorial"
   export MODELS_DIR="$HOME/models/isaaclab_arena/sequential_static_manipulation_tutorial"

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


**Prerequisite: launch the GR00T policy server**

The Arena evaluation client runs in the Arena runtime and connects to a GR00T policy server.
Before running the evaluation, open another terminal, separate from the Arena runtime, for the server.
The server runs out of
the `Isaac-GR00T source <https://github.com/NVIDIA/Isaac-GR00T/tree/e29d8fc50b0e4745120ae3fb72447986fe638aa6>`_
submodule pinned at commit ``e29d8fc``; populate it with
``git submodule update --init submodules/Isaac-GR00T`` if it is not already
checked out. Then, from the repository root in that terminal, launch the server and
leave it running:

.. note::

   Blackwell GPUs with compute capability ``sm_120`` require CUDA 12.8 or newer. The
   `GR00T environment setup guide <https://github.com/NVIDIA/Isaac-GR00T/blob/e29d8fc50b0e4745120ae3fb72447986fe638aa6/README.md#set-up-the-environment>`_
   specifies CUDA 12.8 and ``pytorch-cu128`` for RTX 5090 systems.

.. code-block:: bash

   cd submodules/Isaac-GR00T
   uv run python gr00t/eval/run_gr00t_server.py \
     --modality-config-path ../../isaaclab_arena_policies/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
     --model-path $HOME/models/isaaclab_arena/sequential_static_manipulation_tutorial/checkpoint-20000 \
     --embodiment-tag GR1 \
     --device cuda --host 127.0.0.1 --port 5555


Step 1: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first run the policy in a single environment with visualization via the GUI.

The Arena GR00T evaluation client is configured by a config file at ``isaaclab_arena_policies/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml``.

.. dropdown:: Configuration file (``gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml``):
   :animate: fade-in

   .. code-block:: yaml

      language_instruction: "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
      action_horizon: 16
      video_backend: decord
      modality_config_path: isaaclab_arena_policies/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py

      action_chunk_length: 16
      gr00t_embodiment_adapter: gr1_arms

      pov_cam_name_sim: "robot_pov_cam_rgb"

      original_image_size: [512, 512, 3]
      target_image_size: [512, 512, 3]


Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
     --policy_config_yaml_path isaaclab_arena_policies/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
     --remote_host 127.0.0.1 \
     --remote_port 5555 \
     --num_steps 2000 \
     --enable_cameras \
     put_item_in_fridge_and_close_door \
     --embodiment gr1_joint \
     --object ranch_dressing_hope_robolab

At the end of the evaluation, the policy runner writes the evaluation report, which shows the number of finished
episodes and the success rate for this sequential task.
Each episode's subtask completion is recorded in the ``progress`` field of its episode results record.
You should see similar results. The success rate shall be greater than 0.9, and the number of episodes should be in the range of 3-6.

Note that the success rate is computed over the entire evaluation process, and is affected by the quality of
post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. tab-set::

   .. tab-item:: Best Quality

      5 episodes with a success rate of 1.0.

   .. tab-item:: Low Hardware Requirements

      Evaluated with checkpoint-30000, instead of checkpoint-20000 referenced in the policy configuration file.
      4 episodes with a success rate of 0.75.

Step 2: Run Parallel environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple parallel environments is also supported by the policy runner.

Test the policy in 10 parallel environments with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
     --policy_config_yaml_path isaaclab_arena_policies/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
     --remote_host 127.0.0.1 \
     --remote_port 5555 \
     --num_steps 2000 \
     --num_envs 10 \
     --enable_cameras \
     put_item_in_fridge_and_close_door \
     --embodiment gr1_joint \
     --object ranch_dressing_hope_robolab


And during the evaluation, you should see the following output on the console at the end of the evaluation
indicating which environments are terminated (task-specific conditions like the microwave door is opened),
or truncated (if timeouts are enabled, like the maximum episode length is exceeded).

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([7], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of the evaluation, the evaluation report shows 50 finished episodes with a success rate of 0.98.
The success rate might not be 1.0 as more trials are being evaluated, and the number of episodes is more
than the single environment evaluation because of the parallel evaluation.

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

Test the policy in 10 parallel environments with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
   --viz kit \
   --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
   --policy_config_yaml_path isaaclab_arena_policies/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
   --remote_host 127.0.0.1 \
   --remote_port 5555 \
   --num_steps 2000 \
   --num_envs 10 \
   --enable_cameras \
   put_item_in_fridge_and_close_door \
   --embodiment gr1_joint \
   --object_set ketchup_bottle_hope_robolab ranch_dressing_hope_robolab bbq_sauce_bottle_hope_robolab mayonnaise_bottle_hope_robolab


Each environment has a different object spawned from the object set. The same policy is used for all those environments.
At the end of the evaluation, the evaluation report shows 30 finished episodes with a success rate of 0.87.

The policy demonstrates robust intra-class generalization, achieving an average success rate of 86.67%
when evaluated on structurally similar sauce bottles.
While this represents a slight performance regression compared to the training object,
it indicates that the model has successfully learned geometric features common to the category rather than
just over-fitting to a single object mesh.


Step 4: Sequential Batch Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sequential batch of runs, i.e. different tasks, objects, embodiments or policies, can be evaluated by the ``experiment_runner.py`` script.
It minimizes the overhead of reloading system modules and environment classes for each run while keeping the simulation application alive.
The batch is defined in an Experiment YAML file. The example below evaluates the policy on an unseen jug and on the ranch dressing bottle.

.. dropdown:: Configuration file (``gr1_sequential_static_manip_gr00t_experiment.yaml``):
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/gr1_sequential_static_manip_gr00t_experiment.yaml
      :language: yaml

Run the batch evaluation:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/gr1_sequential_static_manip_gr00t_experiment.yaml

This will automatically evaluate the policy with the given configuration.
You should see the following output on the console with the runs and their success rates.

.. code-block:: text

   +--------------------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+-------------------+--------------+-----------------------+
   |                        Run Name                        |   Status  |       Policy Type       | Num Envs | Num Steps | Num Episodes | Num Rebuilds | Recorded Episodes | Success Rate | Subtask Success Rates |
   +--------------------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+-------------------+--------------+-----------------------+
   |          gr1_put_jug_in_fridge_and_close_door          | completed | gr00t_remote_closedloop |    10    |    500    |     None     |      1       |         10        |    0.1000    |     0.2000, 0.2000    |
   | gr1_put_ranch_dressing_bottle_in_fridge_and_close_door | completed | gr00t_remote_closedloop |    10    |    500    |     None     |      1       |         10        |    0.9000    |     0.9000, 0.9000    |
   +--------------------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+-------------------+--------------+-----------------------+

``Subtask Success Rates`` lists each subtask's success rate in order: placing the object, then closing the door.
In a sequential task a subtask only counts once the ones before it are complete. This task also requires both
subtasks to still hold when the episode ends, so its success rate can be lower than the subtask rates.

With the policy trained on using ranch dressing bottle as object of interest,
the success rate for generalizing to putting the unseen object, jug, in the fridge is 0.1.
This is expected as the policy is not trained on the jug, comparing to the success rate of 0.9 for the trained object, ranch dressing bottle.
