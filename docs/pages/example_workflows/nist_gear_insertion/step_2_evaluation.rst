Closed-Loop Policy Inference and Evaluation
-------------------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the models directory if you plan to download pre-trained checkpoints:

.. code:: bash

    export MODELS_DIR=/models/isaaclab_arena/nist_gear_insertion
    mkdir -p $MODELS_DIR

This tutorial assumes you have a trained checkpoint from an external RL Games launcher, or you can
download a pre-trained one as described below.

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   .. code-block:: bash

      hf download \
         nvidia/Arena-Franka-NIST-Gear-Insertion-RL-Task \
         --local-dir $MODELS_DIR/nist_gear_insertion_checkpoint

   After downloading, the checkpoint and matching agent YAML are at:

   ``$MODELS_DIR/nist_gear_insertion_checkpoint/NistGearInsertion_RlGames.pth``

   ``$MODELS_DIR/nist_gear_insertion_checkpoint/agent.yaml``

   Replace checkpoint and agent-config paths in the examples below with these paths.


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
     --viz kit \
     --policy_type rl_games \
     --num_episodes 20 \
     --checkpoint_path $MODELS_DIR/nist_gear_insertion_checkpoint/NistGearInsertion_RlGames.pth \
     --agent_cfg_path $MODELS_DIR/nist_gear_insertion_checkpoint/agent.yaml \
     nist_assembled_gear_mesh_osc

.. note::

   If you use a checkpoint from an external launcher, the checkpoint path is typically in the
   ``logs/rl_games/NistGearInsertionOscRlg/`` directory. Replace the checkpoint path and
   ``--agent_cfg_path`` with the checkpoint and ``params/agent.yaml`` from that run.

Policy-specific arguments (``--policy_type``, ``--checkpoint_path``, ``--agent_cfg_path``, etc.) must come **before** the
environment name. Environment-specific arguments (``--disable_success_termination``, etc.) must come
**after** it.

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'success_rate': 0.95, 'num_episodes': 20}


Method 2: Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more statistically significant results, run across many environments in parallel:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rl_games \
     --num_episodes 1024 \
     --num_envs 64 \
     --env_spacing 2.5 \
     --viz kit \
     --checkpoint_path $MODELS_DIR/nist_gear_insertion_checkpoint/NistGearInsertion_RlGames.pth \
     --agent_cfg_path $MODELS_DIR/nist_gear_insertion_checkpoint/agent.yaml \
     nist_assembled_gear_mesh_osc

.. code-block:: text

   Metrics: {'success_rate': 0.95, 'num_episodes': 1024}

.. image:: ../../../images/nist_gear_insertion_parallel.gif
   :align: center
   :height: 400px


Method 3: Batch Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate multiple checkpoints in sequence, use ``eval_runner.py`` with a JSON config.
Here we evaluate checkpoints from an external RL Games run.
The checkpoint path should be replaced with the timestamp of your run in the ``logs/rl_games/NistGearInsertionOscRlg/`` directory.

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
           "checkpoint_path": "logs/rl_games/NistGearInsertionOscRlg/<timestamp>/nn/NistGearInsertionOscRlg.pth",
           "agent_cfg_path": "logs/rl_games/NistGearInsertionOscRlg/<timestamp>/params/agent.yaml"
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
           "checkpoint_path": "logs/rl_games/NistGearInsertionOscRlg/<timestamp>/nn/last_NistGearInsertionOscRlg_ep_<epoch>_rew_<reward>.pth",
           "agent_cfg_path": "logs/rl_games/NistGearInsertionOscRlg/<timestamp>/params/agent.yaml"
         }
       }
     ]
   }

**2. Run**

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --viz kit \
     --eval_jobs_config eval_config.json

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   nist_gear_model_0500:
   num_episodes                         1024
   success_rate                       0.7500

   nist_gear_model_1000:
   num_episodes                         1024
   success_rate                       0.9500
   ======================================================================


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The NIST gear insertion task reports two metrics:

- ``success_rate``: fraction of episodes where the held gear is seated on the peg within tolerance
- ``num_episodes``: total number of completed episodes during the evaluation run

A well-trained policy should reach a high success rate on this task. Results will vary with
hardware, random seed, and randomization settings.

.. note::

   For evaluation, omit ``--disable_success_termination`` (default): success termination stays
   enabled so metrics such as success rate are meaningful.
