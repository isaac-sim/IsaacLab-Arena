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

The evaluation is an :doc:`Arena Experiment <../../concepts/concept_arena_experiments>` with
one Run. It selects the environment, the ``gr1_joint`` embodiment and the ranch dressing bottle, sets
the language instruction the policy was trained with, connects the GR00T client to the server, and
stops after 2000 steps. The ``gr1_arms`` embodiment adapter maps the robot's camera and joints to the
checkpoint's inputs and outputs.

.. dropdown:: Experiment file (``gr1_put_ranch_bottle_in_fridge_gr00t_experiment.yaml``):
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/gr1_put_ranch_bottle_in_fridge_gr00t_experiment.yaml
      :language: yaml
      :lines: 6-

Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/gr1_put_ranch_bottle_in_fridge_gr00t_experiment.yaml

At the end of the evaluation, the console shows the Run status and its metrics: the success rate
for this sequential task, ``object_peak_speed_exceeded_rate`` for the first subtask,
``revolute_joint_moved_rate`` for the second subtask, and the success rate of each subtask.
With the pre-trained checkpoint, the output looks like this:

.. code-block:: text

   +-----------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
   |                    Run Name                   |   Status  |       Policy Type       | Num Envs | Num Steps | Num Episodes | Num Rebuilds |
   +-----------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
   | gr1_put_ranch_bottle_in_fridge_and_close_door | completed | gr00t_remote_closedloop |    1     |    2000   |     None     |      1       |
   +-----------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   gr1_put_ranch_bottle_in_fridge_and_close_door:
     num_episodes                            5
     object_peak_speed_exceeded_rate_subtask_0     1.0000
     revolute_joint_moved_rate_subtask_1     1.0000
     subtask_success_rate           [0.8, 0.8]
     success_rate                       0.8000
   ======================================================================

The metrics cover all episodes that finish within the 2000 steps. They depend on the quality of the
post-trained policy, the quality of the dataset, and the number of steps in the evaluation, and they
vary between evaluations, especially with few episodes.

If you trained with the Low Hardware Requirements option in :doc:`step_4_policy_training`, the final
checkpoint is ``checkpoint-30000``. Pass that directory to ``--model-path`` when you start the server.

Step 2: Run Parallel environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same Experiment can evaluate the policy in several parallel environments. Set the number of
environments on the command line instead of editing the file: ``runs.<Run name>.<field>=<value>``
changes one field of that Run.

Test the policy in 10 parallel environments with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/gr1_put_ranch_bottle_in_fridge_gr00t_experiment.yaml \
     runs.gr1_put_ranch_bottle_in_fridge_and_close_door.environment_builder.num_envs=10


During the evaluation, the console shows which environments reset because their episode ended,
for example because the task finished or the episode ran out of time:

.. code-block:: text

   Resetting policy state for env_ids: tensor([5], device='cuda:0')

At the end of the evaluation, the console shows the metrics. With 10 environments, the same number of
steps covers about ten times as many episodes as the single environment evaluation, so the success
rate is a more reliable estimate.

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   gr1_put_ranch_bottle_in_fridge_and_close_door:
     num_episodes                           46
     object_peak_speed_exceeded_rate_subtask_0     1.0000
     revolute_joint_moved_rate_subtask_1     1.0000
     subtask_success_rate           [0.8260869565217391, 0.8260869565217391]
     success_rate                       0.8261
   ======================================================================

.. note::

   Note that the embodiment used in closed-loop policy inference is ``gr1_joint``, which is different
   from ``gr1_pink`` used in data generation.
   This is because during tele-operation, the robot is controlled via target end-effector poses,
   which are realized by using the PINK IK controller.
   GR00T N1.6 policy is trained on upper body joint positions, so we use
   ``gr1_joint`` for closed-loop policy inference.


Step 3: Evaluate on Several Sauce Bottles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step evaluates the same policy on four sauce bottles: ketchup, ranch dressing, barbecue sauce and
mayonnaise. The :ref:`asset selection <asset-selection-variation>` variation makes the Run build the
scene once per bottle, in list order. All 10 parallel environments of a build use the same bottle,
and each build runs 10 of the 40 episodes.

The variation belongs to the object that the robot puts in the fridge. That object keeps the name
``ranch_dressing_hope_robolab`` from ``environment.object``, while the variation replaces its asset
with each listed bottle in turn.

.. dropdown:: Experiment file (``gr1_put_sauce_bottles_in_fridge_gr00t_experiment.yaml``):
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/gr1_put_sauce_bottles_in_fridge_gr00t_experiment.yaml
      :language: yaml
      :lines: 6-

Test the policy in 10 parallel environments with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/gr1_put_sauce_bottles_in_fridge_gr00t_experiment.yaml

At the end of the evaluation, the console shows the metrics over all 40 episodes, followed by the
success rate for each bottle:

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   gr1_put_sauce_bottles_in_fridge_and_close_door:
     num_episodes                           40
     object_peak_speed_exceeded_rate_subtask_0     1.0000
     revolute_joint_moved_rate_subtask_1     1.0000
     subtask_success_rate           [0.7, 0.7]
     success_rate                       0.6000
   ======================================================================

   +------------------------------------------------+---------------------------------------------+----------------------------------+----------+-----------+---------+--------------+
   |                      Run                       |                  Variation                  |              Value               | Episodes | Successes | Unknown | Success Rate |
   +------------------------------------------------+---------------------------------------------+----------------------------------+----------+-----------+---------+--------------+
   | gr1_put_sauce_bottles_in_fridge_and_close_door | ranch_dressing_hope_robolab.asset_selection | "bbq_sauce_bottle_hope_robolab"  |    10    |     5     |    0    |    50.0%     |
   | gr1_put_sauce_bottles_in_fridge_and_close_door | ranch_dressing_hope_robolab.asset_selection |  "ketchup_bottle_hope_robolab"   |    10    |     6     |    0    |    60.0%     |
   | gr1_put_sauce_bottles_in_fridge_and_close_door | ranch_dressing_hope_robolab.asset_selection | "mayonnaise_bottle_hope_robolab" |    10    |     5     |    0    |    50.0%     |
   | gr1_put_sauce_bottles_in_fridge_and_close_door | ranch_dressing_hope_robolab.asset_selection |  "ranch_dressing_hope_robolab"   |    10    |     8     |    0    |    80.0%     |
   +------------------------------------------------+---------------------------------------------+----------------------------------+----------+-----------+---------+--------------+

The policy was trained only on the ranch dressing bottle and succeeds with it in 8 of 10 episodes.
On each of the other three bottles it succeeds in 5 or 6 of 10 episodes, so it partly transfers to
bottles it has not seen.


Step 4: Sequential Batch Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An Experiment can also hold several Runs. ``experiment_runner.py`` evaluates them one after another
in the same simulation application, and each Run reports its own metrics.
The Experiment below evaluates the policy on an unseen jug and on the ranch dressing bottle as two Runs.

.. dropdown:: Experiment file (``gr1_sequential_static_manip_gr00t_experiment.yaml``):
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/gr1_sequential_static_manip_gr00t_experiment.yaml
      :language: yaml
      :lines: 6-

Run the batch evaluation:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/gr1_sequential_static_manip_gr00t_experiment.yaml

This will automatically evaluate the policy with the given configuration and output the metrics.
You should see the following output on the console indicating the runs and metrics.

.. code-block:: text

   +--------------------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
   |                        Run Name                        |   Status  |       Policy Type       | Num Envs | Num Steps | Num Episodes | Num Rebuilds |
   +--------------------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
   |          gr1_put_jug_in_fridge_and_close_door          | completed | gr00t_remote_closedloop |    10    |    500    |     None     |      1       |
   | gr1_put_ranch_dressing_bottle_in_fridge_and_close_door | completed | gr00t_remote_closedloop |    10    |    500    |     None     |      1       |
   +--------------------------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   gr1_put_jug_in_fridge_and_close_door:
     num_episodes                           10
     object_peak_speed_exceeded_rate_subtask_0     1.0000
     revolute_joint_moved_rate_subtask_1     1.0000
     subtask_success_rate           [0.2, 0.2]
     success_rate                       0.1000

   gr1_put_ranch_dressing_bottle_in_fridge_and_close_door:
     num_episodes                           10
     object_peak_speed_exceeded_rate_subtask_0     1.0000
     revolute_joint_moved_rate_subtask_1     1.0000
     subtask_success_rate           [0.9, 0.9]
     success_rate                       0.9000
   ======================================================================

With the policy trained on using ranch dressing bottle as object of interest,
the success rate for generalizing to putting the unseen object, jug, in the fridge is 0.1.
This is expected as the policy is not trained on the jug, comparing to the success rate of 0.9 for the trained object, ranch dressing bottle.
