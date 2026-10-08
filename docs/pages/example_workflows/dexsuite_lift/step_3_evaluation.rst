Evaluation in Arena
--------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the models directory:

.. code-block:: bash

   export MODELS_DIR=/models/isaaclab_arena/dexsuite_lift
   mkdir -p $MODELS_DIR

This step evaluates Isaac Lab's published Newton state-policy checkpoint using
Arena's ``dexsuite_lift`` environment. Arena uses the corresponding policy,
observation, command, control-rate, and physics configuration with its
procedural cube and pose-range reset.

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

      mkdir -p "$MODELS_DIR/params"
      cp isaaclab_arena_examples/policy/dexsuite_lift_agent.yaml \
        "$MODELS_DIR/params/agent.yaml"

   After downloading, the checkpoint is at:

   ``$MODELS_DIR/Isaac-Lift-KukaAllegro.pt``

   Published checkpoints do not include ``params/agent.yaml``. The copy command
   installs Arena's checked-in snapshot of Isaac Lab's state-policy runner
   configuration where ``RslRlActionPolicy`` expects it.

.. note::

   If you trained locally (see :doc:`step_2_policy_training`), your checkpoints
   are at:

   ``logs/rsl_rl/lift_kuka_allegro/<timestamp>/model_<iter>.pt``

   Replace the checkpoint paths in the examples below accordingly.


Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   PYOPENGL_PLATFORM=glx python isaaclab_arena/evaluation/policy_runner.py \
     --viz newton_gl \
     --policy_type rsl_rl \
     --num_episodes 20 \
     --checkpoint_path $MODELS_DIR/Isaac-Lift-KukaAllegro.pt \
     dexsuite_lift

At the end of the run, metrics are printed to the console:

.. code-block:: text

   Metrics: {'num_episodes': 20, 'success_rate': 0.75}


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

   PYOPENGL_PLATFORM=glx python isaaclab_arena/evaluation/policy_runner.py \
     --viz newton_gl \
     --policy_type rsl_rl \
     --num_episodes 400 \
     --num_envs 64 \
     --env_spacing 3 \
     --checkpoint_path $MODELS_DIR/Isaac-Lift-KukaAllegro.pt \
     dexsuite_lift

.. code-block:: text

   Metrics: {'num_episodes': 400, 'success_rate': 0.82}


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``dexsuite_lift`` task reports:

- ``success_rate``: fraction of episodes where the object reached the target
  position within 5 cm tolerance.
- ``num_episodes``: total number of completed episodes.
