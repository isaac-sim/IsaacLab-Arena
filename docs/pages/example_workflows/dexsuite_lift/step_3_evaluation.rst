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

   The Experiment YAML below reads the checkpoint from this path.

.. note::

   If you trained locally (see :doc:`step_2_policy_training`), your checkpoints
   are at:

   ``logs/rsl_rl/lift_kuka_allegro/<timestamp>/model_<iter>.pt``

   To evaluate one of them, add this override to the commands below:

   ``runs.dexsuite_lift_rsl_rl.policy.checkpoint_path=logs/rsl_rl/lift_kuka_allegro/<timestamp>/model_<iter>.pt``


Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This Experiment YAML declares one Run. It evaluates the downloaded checkpoint with the ``rsl_rl``
policy in the ``dexsuite_lift`` environment for 20 episodes:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/reinforcement_learning/dexsuite_lift_rsl_rl_experiment.yaml
   :language: yaml
   :start-at: runs:

Run it with the Experiment Runner:

.. code-block:: bash

   PYOPENGL_PLATFORM=glx python isaaclab_arena/evaluation/experiment_runner.py \
     --viz newton_gl \
     --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/dexsuite_lift_rsl_rl_experiment.yaml

When the Run finishes, the runner prints a metrics summary:

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   dexsuite_lift_rsl_rl:
     num_episodes                           20
     success_rate                       0.7500
   ======================================================================


.. image:: ../../../images/dexsuite_lift_task.gif
   :align: center
   :height: 400px


.. tip::

   You can also evaluate a Newton-trained model using PhysX. These overrides select PhysX and
   replace the 20-episode limit with an 800-step limit. ``num_steps`` and ``num_episodes`` cannot
   both be set, so the episode limit is set to ``null``:

   .. code-block:: bash

      python isaaclab_arena/evaluation/experiment_runner.py \
        --viz kit \
        --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/dexsuite_lift_rsl_rl_experiment.yaml \
        runs.dexsuite_lift_rsl_rl.environment_builder.presets=PHYSX \
        runs.dexsuite_lift_rsl_rl.rollout_limit.num_episodes=null \
        runs.dexsuite_lift_rsl_rl.rollout_limit.num_steps=800

   However, the model behaviour may differ significantly when training and
   evaluation use different physics backends; the published policy is validated
   against Newton.


Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For statistically significant results, run across many environments in parallel. Use the same
Experiment YAML and override the number of environments, their spacing, and the number of
episodes on the command line:

.. code-block:: bash

   PYOPENGL_PLATFORM=glx python isaaclab_arena/evaluation/experiment_runner.py \
     --viz newton_gl \
     --experiment_config isaaclab_arena_environments/experiment_configs/reinforcement_learning/dexsuite_lift_rsl_rl_experiment.yaml \
     runs.dexsuite_lift_rsl_rl.environment_builder.num_envs=64 \
     runs.dexsuite_lift_rsl_rl.environment_builder.env_spacing=3 \
     runs.dexsuite_lift_rsl_rl.rollout_limit.num_episodes=400

.. code-block:: text

   ======================================================================
   METRICS SUMMARY
   ======================================================================

   dexsuite_lift_rsl_rl:
     num_episodes                          400
     success_rate                       0.8200
   ======================================================================


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``dexsuite_lift`` task reports:

- ``success_rate``: fraction of episodes where the object reached the target
  position within 5 cm tolerance.
- ``num_episodes``: total number of completed episodes.
