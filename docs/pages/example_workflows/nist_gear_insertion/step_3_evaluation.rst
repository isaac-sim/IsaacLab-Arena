Closed-Loop Policy Inference and Evaluation
-------------------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the models directory if you plan to organize checkpoints locally:

.. code-block:: bash

    export MODELS_DIR=models/isaaclab_arena/nist_gear_insertion
    mkdir -p $MODELS_DIR

This tutorial assumes you've completed :doc:`step_2_policy_training` and have a trained checkpoint,
or you can place a checkpoint in ``$MODELS_DIR`` and use it in the commands below.

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   .. code-block:: bash

      hf download \
         nvidia/Arena-NIST-Gear-Insertion-RL-Task \
         --local-dir $MODELS_DIR/nist_gear_insertion_checkpoint

   After downloading, the checkpoint is at:

   ``$MODELS_DIR/nist_gear_insertion_checkpoint/best_NistGearInsertionOscRlg.pth``

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
     --checkpoint_path $MODELS_DIR/<checkpoint>.pth \
     --agent_cfg_path isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml \
     --policy_type rl_games \
     --num_episodes 20 \
     --num_envs 1 \
     --visualizer kit \
     nist_assembled_gear_mesh_osc

.. note::

   If you trained the model yourself, replace the checkpoint path with the path to your own checkpoint.

Policy-specific arguments (``--policy_type``, ``--checkpoint_path``, ``--agent_cfg_path``, etc.)
must come **before** the environment name. Environment-specific arguments must come **after** it.

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'success_rate': 0.99, 'num_episodes': 1024}

.. image:: ../../../images/nist_gear_insertion_task.gif
   :align: center
   :height: 400px


Method 2: Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more statistically significant results, run across many environments in parallel.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --checkpoint_path $MODELS_DIR/<checkpoint>.pth \
     --agent_cfg_path isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml \
     --policy_type rl_games \
     --num_episodes 1024 \
     --num_envs 64 \
     --env_spacing 2.5 \
     --visualizer kit \
     nist_assembled_gear_mesh_osc

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'success_rate': 0.99, 'num_episodes': 1024}

.. image:: ../../../images/nist_gear_insertion_parallel.gif
   :align: center
   :height: 400px


Method 3: Batch Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate multiple checkpoints in sequence, use ``eval_runner.py`` with a JSON config.
Here we evaluate checkpoints from the same training run.
The checkpoint paths should be replaced with the timestamped run directory in
``logs/rl_games/NistGearInsertionOscRlg/``.

**1. Create an evaluation config**

Create a file ``eval_config.json``:

.. code-block:: json

   {
     "jobs": [
       {
         "name": "nist_gear_model_0500",
         "policy_type": "rl_games",
         "num_episodes": 1024,
         "arena_env_args": {
           "environment": "nist_assembled_gear_mesh_osc",
           "num_envs": 64
         },
         "policy_config_dict": {
           "checkpoint_path": "models/isaaclab_arena/nist_gear_insertion/model_0500.pth",
           "agent_cfg_path": "isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml"
         }
       },
       {
         "name": "nist_gear_model_1000",
         "policy_type": "rl_games",
         "num_episodes": 1024,
         "arena_env_args": {
           "environment": "nist_assembled_gear_mesh_osc",
           "num_envs": 64
         },
         "policy_config_dict": {
           "checkpoint_path": "models/isaaclab_arena/nist_gear_insertion/model_1000.pth",
           "agent_cfg_path": "isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml"
         }
       }
     ]
   }

**2. Run**

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --eval_jobs_config eval_config.json

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   nist_gear_model_0500:
   num_episodes                         1024
   success_rate                       0.95

   nist_gear_model_1000:
   num_episodes                         1024
   success_rate                       0.9900
   ======================================================================


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

The NIST gear insertion task reports two metrics:

- ``success_rate``: fraction of episodes where the gear is successfully inserted
- ``num_episodes``: number of completed episodes in the evaluation run

A well-trained policy should show improving insertion performance as training progresses.
Results will vary with hardware, random seed, and randomization settings.

.. note::

   For evaluation, omit ``--rl_training_mode`` (default): success termination stays enabled so
   metrics such as success rate are meaningful. For RL Games training, pass ``--rl_training_mode`` to
   disable success termination.
