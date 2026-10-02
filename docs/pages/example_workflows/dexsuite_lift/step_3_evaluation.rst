Evaluation in Arena
--------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the models directory:

.. code-block:: bash

   export MODELS_DIR=/models/isaaclab_arena/dexsuite_lift
   mkdir -p $MODELS_DIR

This step evaluates Isaac Lab's published Newton state-policy checkpoint using
Arena's ``dexsuite_lift`` environment. Arena mirrors the corresponding Isaac
Lab play configuration.

.. dropdown:: Download Pre-trained Model (skip training)
   :animate: fade-in

   .. code-block:: bash

      /isaac-sim/python.sh - <<'PY'
      import os
      import shutil

      from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

      source = get_published_pretrained_checkpoint(
          "rsl_rl", "Isaac-Lift-KukaAllegro", "newtonmjwarp", "none"
      )
      assert source is not None
      destination = os.path.join(os.environ["MODELS_DIR"], "Isaac-Lift-KukaAllegro.pt")
      shutil.copy2(source, destination)
      print(destination)
      PY

   After downloading, the checkpoint is at:

   ``$MODELS_DIR/Isaac-Lift-KukaAllegro.pt``

   Published checkpoints do not include ``params/agent.yaml``. Arena therefore
   loads the environment's registered ``KukaAllegroPPORunnerCfg``.

.. note::

   If you trained locally (see :doc:`step_2_policy_training`), your checkpoints
   are at:

   ``logs/rsl_rl/lift_kuka_allegro/<timestamp>/model_<iter>.pt``

   Replace the checkpoint paths in the examples below accordingly.


Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --num_episodes 100 \
     --num_envs 1 \
     --checkpoint_path $MODELS_DIR/Isaac-Lift-KukaAllegro.pt \
     dexsuite_lift

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'num_episodes': 100, 'success_rate': 0.98}

The same checkpoint achieved 96 successes over 100 sequential episodes in
Isaac Lab's native play environment. The two measurements agree within normal
sampling variation.

Use one environment for this comparison. Successful Arena episodes terminate
early, whereas Isaac Lab runs them until timeout; stopping a many-environment
run after the first 100 completions would therefore over-sample shorter,
successful episodes.

To inspect the rollout interactively:

.. code-block:: bash

   PYOPENGL_PLATFORM=glx python isaaclab_arena/evaluation/policy_runner.py \
     --viz newton_gl \
     --policy_type rsl_rl \
     --num_episodes 5 \
     --checkpoint_path $MODELS_DIR/Isaac-Lift-KukaAllegro.pt \
     dexsuite_lift


.. image:: ../../../images/dexsuite_lift_task.gif
   :align: center
   :height: 400px


.. tip::

   You can also evaluate a Newton-trained model using PhysX:

   .. code-block:: bash

      python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --presets physx \
        --policy_type rsl_rl \
        --num_steps 800 \
        --checkpoint_path $MODELS_DIR/Isaac-Lift-KukaAllegro.pt \
        dexsuite_lift

   However, the model behaviour may differ significantly when training and
   evaluation use different physics backends; the published policy is validated
   against Newton.


Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For statistically significant results, run across many environments in parallel:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --num_steps 5000 \
     --num_envs 64 \
     --env_spacing 3 \
     --checkpoint_path $MODELS_DIR/Isaac-Lift-KukaAllegro.pt \
     dexsuite_lift

Use a fixed-step parallel rollout for throughput checks. Use the single-
environment command above when comparing an exact episode count with Isaac Lab.


Batch Evaluation
^^^^^^^^^^^^^^^^

To evaluate multiple checkpoints in sequence, use ``experiment_runner.py`` with a
JSON config.

**1. Create an evaluation config**

Create a file ``eval_config.json``:

.. code-block:: json

   {
     "jobs": [
       {
         "name": "dexsuite_lift_7500",
         "arena_env_args": {
           "environment": "dexsuite_lift",
           "num_envs": 64,
           "env_spacing": 3
         },
         "num_steps": 5000,
         "policy_type": "rsl_rl",
         "policy_config_dict": {
           "checkpoint_path": "models/isaaclab_arena/dexsuite_lift/model_7500.pt"
         }
       },
       {
         "name": "dexsuite_lift_14999",
         "arena_env_args": {
           "environment": "dexsuite_lift",
           "num_envs": 64,
           "env_spacing": 3
         },
         "num_steps": 5000,
         "policy_type": "rsl_rl",
         "policy_config_dict": {
           "checkpoint_path": "models/isaaclab_arena/dexsuite_lift/model_14999.pt"
         }
       }
     ]
   }

**2. Run**

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py --eval_jobs_config eval_config.json


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``dexsuite_lift`` task reports:

- ``success_rate``: fraction of episodes where the object reached the target
  position within 5 cm tolerance.
- ``num_episodes``: total number of completed episodes.
