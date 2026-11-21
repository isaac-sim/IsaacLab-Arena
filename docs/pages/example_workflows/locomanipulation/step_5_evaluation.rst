Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.6 policy in closed-loop
and evaluating it in Arena G1 Loco Manipulation Task environment.


:runtime_setup_tabs:`--group gr00t-client`

Once the runtime is ready, set the dataset and models directories.

.. code-block:: bash

   export DATASET_DIR="$HOME/datasets/isaaclab_arena/locomanipulation_tutorial"
   export MODELS_DIR="$HOME/models/isaaclab_arena/locomanipulation_tutorial"

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_4_policy_training>` or downloaded the
pre-trained model checkpoint below:

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   These commands can be used to download the pre-trained GR00T N1.6 policy checkpoint,
   such that the preceding steps can be skipped.
   This step requires the Hugging Face CLI, which can be installed by following the
   `official instructions <https://huggingface.co/docs/huggingface_hub/installation>`_.

   To download run:

   .. code-block:: bash

      hf download \
         --revision gn1_6 \
         nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
         --local-dir $MODELS_DIR/checkpoint-20000


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
     --modality-config-path ../../isaaclab_arena_policies/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py \
     --model-path $HOME/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000 \
     --embodiment-tag NEW_EMBODIMENT \
     --device cuda --host 127.0.0.1 --port 5555


Step 1: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first run the policy in a single environment with visualization via the GUI.

The evaluation is an :doc:`Arena Experiment <../../concepts/concept_arena_experiments>` with
one Run. It selects the environment, the ``g1_wbc_joint`` embodiment and the brown box, connects the
GR00T client to the server, and stops after 1500 steps. An episode times out after 30 seconds, which
is 1500 steps. The ``g1_wbc`` embodiment adapter maps the robot's camera and joints to the
checkpoint's inputs and outputs.

.. dropdown:: Experiment file (``g1_locomanip_gr00t_experiment.yaml``):
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/g1_locomanip_gr00t_experiment.yaml
      :language: yaml
      :lines: 6-

Test the policy in a single environment with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/g1_locomanip_gr00t_experiment.yaml

At the end of the evaluation, the console shows the Run status and its metrics: the success rate,
``object_peak_speed_exceeded_rate`` and the number of episodes. With the pre-trained checkpoint, the
output looks like this:

.. code-block:: text

   +-----------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
   |           Run Name          |   Status  |       Policy Type       | Num Envs | Num Steps | Num Episodes | Num Rebuilds |
   +-----------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
   | g1_pick_and_place_brown_box | completed | gr00t_remote_closedloop |    1     |    1500   |     None     |      1       |
   +-----------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   g1_pick_and_place_brown_box:
     num_episodes                            1
     object_peak_speed_exceeded_rate     1.0000
     success_rate                       1.0000
   ======================================================================

The metrics cover all episodes that finish within the 1500 steps. They depend on the quality of the
post-trained policy, the quality of the dataset, and the number of steps in the evaluation, and they
vary between evaluations, especially with few episodes.

Step 2: Run Parallel Environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same Experiment can evaluate the policy in several parallel environments. Set the number of
environments on the command line instead of editing the file: ``runs.<Run name>.<field>=<value>``
changes one field of that Run.

Test the policy in 5 parallel environments with visualization via the GUI run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/g1_locomanip_gr00t_experiment.yaml \
     runs.g1_pick_and_place_brown_box.environment_builder.num_envs=5


During the evaluation, the console shows which environments reset because their episode ended,
for example because the brown box was placed in the blue bin or the episode ran out of time:

.. code-block:: text

   Resetting policy state for env_ids: tensor([3], device='cuda:0')

At the end of the evaluation, the console shows the metrics. An episode ends after 1500 steps at
the latest, so every environment finishes at least one episode and the success rate also counts the
attempts that time out. Environments that finish early start a new episode, which counts only if it
also ends within the 1500 steps.

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   g1_pick_and_place_brown_box:
     num_episodes                            6
     object_peak_speed_exceeded_rate     1.0000
     success_rate                       0.5000
   ======================================================================

.. note::

   Note that the embodiment used in closed-loop policy inference is ``g1_wbc_joint``, which is different
   from ``g1_wbc_pink`` used in data generation.
   This is because during tele-operation, the upper body is controlled via target end-effector poses,
   which are realized by using the PINK IK controller, and the lower body is controlled via a WBC policy.
   GR00T N1.6 policy is trained on upper body joint positions and lower body WBC policy inputs, so we use
   ``g1_wbc_joint`` for closed-loop policy inference.

.. note::

   The training dataset was generated with CPU physics, while the commands above run physics on the GPU,
   the default device. To evaluate with CPU physics instead, add ``--device cpu``.
