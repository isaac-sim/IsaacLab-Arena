Policy Training (Isaac Lab)
----------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

.. important::

   Training is performed **in Isaac Lab** (not Arena). The command below keeps
   Isaac Lab's 4,096-environment and multi-shape defaults. Arena evaluation
   mirrors the same play configuration and Newton backend.


Training Command
^^^^^^^^^^^^^^^^

Train the ``Isaac-Lift-KukaAllegro`` task with Isaac Lab's unified CLI:

.. code-block:: bash

   uv run isaaclab train \
     --rl_library rsl_rl \
     --task Isaac-Lift-KukaAllegro \
     physics=newton_mjwarp

``physics=newton_mjwarp`` explicitly selects the backend used by the published
checkpoint. The task's default object preset samples the full training shape
set; do not add ``presets=cube`` when reproducing that checkpoint.

This uses the ``KukaAllegroPPORunnerCfg`` configuration defined in
Isaac Lab, which provides:

- **Actor/Critic**: MLP [512, 256, 128], ELU activation, observation normalization enabled
- **Observation groups**: ``policy`` + ``proprio`` + ``perception`` (all three groups
  concatenated, each with 5-step history)
- **Algorithm**: PPO with adaptive learning rate schedule, starting at ``1e-3``
- **Training**: 15,000 iterations, 32 steps per environment, 4,096 parallel environments
- **Physics**: Newton (MuJoCo-Warp solver) when ``physics=newton_mjwarp`` is used

Checkpoints are saved every 250 iterations to
``logs/rsl_rl/lift_kuka_allegro/<timestamp>/``.


Overriding Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Hyperparameters can be overridden with Hydra-style CLI arguments:

.. code-block:: bash

   uv run isaaclab train \
     --rl_library rsl_rl \
     --task Isaac-Lift-KukaAllegro \
     physics=newton_mjwarp \
     agent.max_iterations=20000 agent.save_interval=500 agent.algorithm.learning_rate=0.0005


Resuming from a Checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^

Resume from the newest compatible local run with ``--checkpoint latest``, or
replace ``latest`` with an explicit checkpoint path:

.. code-block:: bash

   uv run isaaclab train \
     --rl_library rsl_rl \
     --task Isaac-Lift-KukaAllegro \
     --checkpoint latest \
     physics=newton_mjwarp


Monitoring Training
^^^^^^^^^^^^^^^^^^^

Launch Tensorboard to monitor progress:

.. code-block:: bash

   python -m tensorboard.main --logdir logs/rsl_rl

During training, each iteration prints a summary to the console showing rewards,
losses, and termination statistics.


Expected Results
^^^^^^^^^^^^^^^^

After 15,000 iterations, the Kuka Allegro hand should reliably grasp and lift
the sampled objects to target poses.

.. note::

   Training performance depends on hardware, random seed, and physics
   configuration. Newton training may be slower than PhysX due to the more
   accurate contact solver. For best results, use a powerful GPU (e.g., RTX
   4090, A100, L40).
