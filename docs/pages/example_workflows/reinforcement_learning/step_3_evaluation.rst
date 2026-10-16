Closed-Loop Policy Inference and Evaluation
-------------------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the models directory if you plan to download pre-trained checkpoints:

.. code:: bash

    export MODELS_DIR=/models/isaaclab_arena/reinforcement_learning
    mkdir -p $MODELS_DIR

This tutorial assumes you've completed :doc:`step_2_policy_training` and have a trained checkpoint,
or you can download a pre-trained one as described below.

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   .. code-block:: bash

      hf download \
         nvidia/Arena-Franka-Lift-Object-RL-Task \
         --local-dir $MODELS_DIR/lift_object_checkpoint

   After downloading, the checkpoint is at:

   ``$MODELS_DIR/lift_object_checkpoint/model_1999.pt``

   The same folder also contains ``model_1000.pt``. The Experiment YAMLs below read
   checkpoints from this folder.


Evaluation Methods
^^^^^^^^^^^^^^^^^^

There are three ways to evaluate a trained policy. Each one runs ``experiment_runner.py`` with an
Experiment YAML:

1. **Single environment**: detailed evaluation with metrics
2. **Parallel environments**: larger-scale statistical evaluation
3. **Batch evaluation**: automated evaluation across multiple checkpoints


Method 1: Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This Experiment YAML declares one Run. It evaluates the downloaded ``model_1999.pt`` checkpoint
with the ``rsl_rl`` policy in the ``lift_object`` environment for 20 episodes:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/reinforcement_learning/lift_object_rsl_rl_experiment.yaml
   :language: yaml
   :start-at: runs:

Run it with the Experiment Runner:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/lift_object_rsl_rl_experiment.yaml

.. note::

   If you train the model yourself, the checkpoint path is typically in the ``logs/rsl_rl/generic_experiment/`` directory.
   Override the checkpoint path to evaluate your own checkpoint. Replace ``<timestamp>`` with the
   folder name of your training run:

   .. code-block:: bash

      python isaaclab_arena/evaluation/experiment_runner.py \
        --viz kit \
        --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/lift_object_rsl_rl_experiment.yaml \
        runs.lift_object_rsl_rl.policy.checkpoint_path=logs/rsl_rl/generic_experiment/<timestamp>/model_1999.pt

Policy settings such as ``checkpoint_path`` go under ``policy`` in the Run. Environment settings
such as ``object`` or ``embodiment`` go under ``environment``.

When the Run finishes, the runner prints a metrics summary:

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   lift_object_rsl_rl:
     num_episodes                           20
     success_rate                       0.8100
   ======================================================================


Method 2: Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more statistically significant results, run across many environments in parallel. Use the
same Experiment YAML and override the number of environments, their spacing, and the number of
episodes on the command line:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/lift_object_rsl_rl_experiment.yaml \
     runs.lift_object_rsl_rl.environment_builder.num_envs=64 \
     runs.lift_object_rsl_rl.environment_builder.env_spacing=2.5 \
     runs.lift_object_rsl_rl.rollout_limit.num_episodes=1024

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   lift_object_rsl_rl:
     num_episodes                         1024
     success_rate                       0.7200
   ======================================================================

.. image:: ../../../images/lift_object_rl_parallel.gif
   :align: center
   :height: 400px


Method 3: Batch Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate multiple checkpoints in sequence, declare one Run per checkpoint in an Experiment YAML.
This YAML evaluates ``model_1000.pt`` and ``model_1999.pt`` from the downloaded folder. Values
under ``shared`` apply to both Runs, so each Run only sets its checkpoint:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/reinforcement_learning/lift_object_rsl_rl_checkpoints_experiment.yaml
   :language: yaml
   :start-at: shared:

Run both checkpoints with one command:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/lift_object_rsl_rl_checkpoints_experiment.yaml

To evaluate the models you trained yourself, override the checkpoint path of each Run. Replace
``<timestamp>`` with the folder name of your training run in the ``logs/rsl_rl/generic_experiment/``
directory:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/lift_object_rsl_rl_checkpoints_experiment.yaml \
     runs.lift_object_model_1000.policy.checkpoint_path=logs/rsl_rl/generic_experiment/<timestamp>/model_1000.pt \
     runs.lift_object_model_1999.policy.checkpoint_path=logs/rsl_rl/generic_experiment/<timestamp>/model_1999.pt

The runner evaluates the Runs in YAML order and prints a summary for both:

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   lift_object_model_1000:
     num_episodes                         1024
     success_rate                       0.6526

   lift_object_model_1999:
     num_episodes                         1024
     success_rate                       0.7408
   ======================================================================


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Lift Object task reports two metrics:

- ``success_rate``: fraction of episodes where the object reached the target position within tolerance
- ``num_episodes``: total number of completed episodes during the evaluation run

A well-trained policy should reach 70–90% success rate. Results will vary with the target range,
random seed, and hardware.

.. note::

   For evaluation, leave ``environment.rl_training_mode`` at its default, ``false``: success
   termination stays enabled so metrics such as success rate are meaningful. For RSL-RL training,
   pass ``--rl_training_mode`` to disable success termination.
