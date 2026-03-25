Closed-Loop Policy Inference and Evaluation
-------------------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

Once inside the container, set the models directory if you plan to download pre-trained checkpoints:

.. code:: bash

    export MODELS_DIR=models/isaaclab_arena/reinforcement_learning
    mkdir -p $MODELS_DIR

This tutorial assumes you've completed :doc:`step_2_policy_training` and have a trained checkpoint,
or you can download a pre-trained one as described below.

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   .. code-block:: bash

      hf download \
         nvidia/IsaacLab-Arena-Lift-Object-RL \
         model_11999.pt \
         --local-dir $MODELS_DIR/lift_object_checkpoint

   After downloading, the checkpoint is at:

   ``$MODELS_DIR/lift_object_checkpoint/model_11999.pt``

   Replace checkpoint paths in the examples below with this path.


Evaluation Methods
^^^^^^^^^^^^^^^^^^

There are three ways to evaluate a trained policy:

1. **Single environment** (``policy_runner.py``): detailed evaluation with metrics
2. **Parallel environments** (``policy_runner.py``): larger-scale statistical evaluation
3. **Batch evaluation** (``eval_runner.py``): automated evaluation across multiple checkpoints


Method 1: Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --num_steps 1000 \
     --checkpoint_path logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_11999.pt \
     lift_object

.. note::

   If you downloaded the pre-trained model from Hugging Face, replace the checkpoint path with:
   ``$MODELS_DIR/lift_object_checkpoint/model_11999.pt``

Policy-specific arguments (``--policy_type``, ``--checkpoint_path``, etc.) must come **before** the
environment name. Environment-specific arguments (``--object``, ``--embodiment``, etc.) must come
**after** it.

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'success_rate': 0.85, 'num_episodes': 12}


Method 2: Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more statistically significant results, run across many environments in parallel:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --num_steps 5000 \
     --num_envs 64 \
     --checkpoint_path logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_11999.pt \
     --headless \
     lift_object

.. code-block:: text

   Metrics: {'success_rate': 0.83, 'num_episodes': 156}


Method 3: Batch Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate multiple checkpoints in sequence, use ``eval_runner.py`` with a JSON config.

**1. Create an evaluation config**

Create a file ``eval_config.json``:

.. code-block:: json

   {
     "policy_runner_args": {
       "policy_type": "rsl_rl",
       "num_steps": 5000,
       "num_envs": 64,
       "headless": true
     },
     "evaluations": [
       {
         "checkpoint_path": "logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_5999.pt",
         "environment": "lift_object"
       },
       {
         "checkpoint_path": "logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_11999.pt",
         "environment": "lift_object"
       }
     ]
   }

**2. Run**

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py --eval_jobs_config eval_config.json

.. code-block:: text

   Evaluating checkpoint 1/2: model_5999.pt
   Metrics: {'success_rate': 0.72, 'num_episodes': 152}

   Evaluating checkpoint 2/2: model_11999.pt
   Metrics: {'success_rate': 0.85, 'num_episodes': 156}

   Summary:
   ========================================
   model_5999.pt  | Success: 72% | Episodes: 152
   model_11999.pt | Success: 85% | Episodes: 156


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Lift Object task reports two metrics:

- ``success_rate``: fraction of episodes where the object reached the target position within tolerance
- ``num_episodes``: total number of completed episodes during the evaluation run

A well-trained policy should reach 70–90% success rate. Results will vary with the target range,
random seed, and hardware.

.. note::

   For evaluation, omit ``--rl_training_mode`` (default): success termination stays enabled so
   metrics such as success rate are meaningful. For RSL-RL training, pass ``--rl_training_mode`` to
   disable success termination.
