Evaluation in Arena
--------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the models directory:

.. code-block:: bash

   export MODELS_DIR=/models/isaaclab_arena/dexsuite_lift
   mkdir -p $MODELS_DIR

This step evaluates a checkpoint using Arena's ``dexsuite_lift`` environment.
Pass ``--presets newton`` to use Newton physics (recommended when the checkpoint
was trained with Newton).

.. dropdown:: Download Pre-trained Model (skip training)
   :animate: fade-in

   .. code-block:: bash

      hf download \
        nvidia/Arena-Dexsuite-Lift-RL-Newton-Task \
        --local-dir $MODELS_DIR

   After downloading, the checkpoint is at:

   ``$MODELS_DIR/model_14999.pt``

.. note::

   If you trained locally (see :doc:`step_2_policy_training`), your checkpoints
   are at:

   ``logs/rsl_rl/dexsuite_kuka_allegro/<timestamp>/model_<iter>.pt``

   Replace the checkpoint paths in the examples below accordingly.


Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz newton \
     --presets newton \
     --policy_type rsl_rl \
     --num_steps 800 \
     --checkpoint_path $MODELS_DIR/model_14999.pt \
     dexsuite_lift

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'success_rate': 0.75, 'num_episodes': 12}


.. image:: ../../../images/dexsuite_lift_task.gif
   :align: center
   :height: 400px


.. tip::

   You can also evaluate a Newton-trained model using PhysX:

   .. code-block:: bash

      python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type rsl_rl \
        --num_steps 800 \
        --checkpoint_path $MODELS_DIR/model_14999.pt \
        dexsuite_lift

   However, the model behaviour may differ significantly when training and
   evaluation use different physics backends. The above model, which was
   trained with Newton, fails to grasp or lift the cube completely when
   evaluated with PhysX.


Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For statistically significant results, run across many environments in parallel:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --presets newton \
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

To evaluate multiple checkpoints in sequence, use ``eval_runner.py`` with a
JSON config.

**1. Create an evaluation config**

Create a file ``eval_config.json``:

.. code-block:: json

   {
     "policy_runner_args": {
       "presets": "newton",
       "policy_type": "rsl_rl",
       "num_steps": 5000,
       "num_envs": 64,
       "env_spacing": 3
     },
     "evaluations": [
       {
         "checkpoint_path": "models/isaaclab_arena/dexsuite_lift/model_7500.pt",
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

- ``success_rate``: fraction of episodes where the object reached the target
  position within 5 cm tolerance.
- ``num_episodes``: total number of completed episodes.
