Evaluation in Arena (Newton Physics)
-------------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the models directory:

.. code:: bash

    export MODELS_DIR=models/isaaclab_arena/dexsuite_lift
    mkdir -p $MODELS_DIR

This step evaluates a checkpoint using Arena's ``dexsuite_lift`` environment running
under **Newton** physics. You can use either a locally trained checkpoint or download
a pre-trained one from Hugging Face.

.. dropdown:: Download Pre-trained Model (skip training)
   :animate: fade-in

   .. code-block:: bash

      hf download \
         nvidia/IsaacLab-Arena-Dexsuite-Lift-RL \
         --local-dir $MODELS_DIR

   After downloading, the checkpoint is at:

   ``$MODELS_DIR/model_14999.pt``

.. note::

   If you trained locally (see :doc:`step_2_policy_training`), your checkpoints are at:

   ``logs/rsl_rl/dexsuite_kuka_allegro/<timestamp>/model_<iter>.pt``

   Replace the checkpoint paths in the examples below accordingly.


Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --visualizer newton \
     --policy_type rsl_rl \
     --num_steps 800 \
     --checkpoint_path $MODELS_DIR/model_14999.pt \
     dexsuite_lift

.. note::

   - Use ``--visualizer newton`` to launch the Newton (MuJoCo) visualizer.
   - Use ``--env_spacing 3`` to match the Dexsuite training layout.

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'success_rate': 0.75, 'num_episodes': 12}


Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For statistically significant results, run across many environments in parallel:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --num_steps 5000 \
     --num_envs 64 \
     --env_spacing 3 \
     --checkpoint_path $MODELS_DIR/model_14999.pt \
     dexsuite_lift

.. code-block:: text

   Metrics: {'success_rate': 0.72, 'num_episodes': 320}


Batch Evaluation
^^^^^^^^^^^^^^^^

To evaluate multiple checkpoints in sequence, use ``eval_runner.py`` with a JSON config.

**1. Create an evaluation config**

Create a file ``eval_config.json``:

.. code-block:: json

   {
     "policy_runner_args": {
       "policy_type": "rsl_rl",
       "num_steps": 5000,
       "num_envs": 64,
       "env_spacing": 3
     },
     "evaluations": [
       {
         "checkpoint_path": "models/isaaclab_arena/dexsuite_lift/model_7499.pt",
         "environment": "dexsuite_lift"
       },
       {
         "checkpoint_path": "models/isaaclab_arena/dexsuite_lift/model_14999.pt",
         "environment": "dexsuite_lift"
       }
     ]
   }

**2. Run**

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py --eval_jobs_config eval_config.json


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``dexsuite_lift`` task reports:

- ``success_rate``: fraction of episodes where the object reached the target position
  within 5 cm tolerance.
- ``min_goal_distance``: minimum distance (metres) between the object and goal during
  each episode, aggregated as mean / min / max across all episodes.
- ``num_episodes``: total number of completed episodes.
