Policy Training (Isaac Lab, Newton)
------------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

.. important::

   Training is performed **in Isaac Lab** (not Arena) using **Newton** physics.
   Add ``presets=newton`` to the training command to switch the physics backend from
   the default PhysX to Newton (MuJoCo-Warp solver). This ensures training and
   evaluation use the same contact model, eliminating the sim-to-sim gap.


Training Command
^^^^^^^^^^^^^^^^

Train the ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` task directly with Isaac Lab's
RSL-RL training script, with Newton physics enabled:

.. code-block:: bash

   python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
     --num_envs 512 \
     presets=newton presets=cube

``presets=newton`` selects the Newton physics backend and ``presets=cube`` switches the
manipulation object to a single-geometry cube (Newton does not support multi-asset
spawning used by the default ``shapes`` preset).

This uses the ``DexsuiteKukaAllegroPPORunnerCfg`` configuration defined in Isaac Lab,
which provides:

- **Actor/Critic**: MLP [512, 256, 128], ELU activation, observation normalization enabled
- **Observation groups**: ``policy`` + ``proprio`` + ``perception`` (all three groups
  concatenated, each with 5-step history)
- **Algorithm**: PPO with adaptive learning rate schedule, starting at ``1e-3``
- **Training**: 15,000 iterations, 32 steps per environment, 512 parallel environments
- **Physics**: Newton (MuJoCo-Warp solver) via ``presets=newton``

Checkpoints are saved every 250 iterations to
``logs/rsl_rl/dexsuite_kuka_allegro/<timestamp>/``.

.. tip::

   Add ``--visualizer newton`` to visualize training with the Newton (MuJoCo) viewer.

.. note::

   Omit ``presets=newton`` to train with PhysX instead.


Overriding Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Hyperparameters can be overridden with Hydra-style CLI arguments:

.. code-block:: bash

   # Change max iterations
   python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
     --num_envs 512 \
     --headless \
     presets=newton presets=cube \
     agent.max_iterations=20000
     # Change save interval
     agent.save_interval=500
     # Change learning rate
     agent.algorithm.learning_rate=0.0005


Monitoring Training
^^^^^^^^^^^^^^^^^^^

Launch Tensorboard to monitor progress:

.. code-block:: bash

   python -m tensorboard.main --logdir logs/rsl_rl

During training, each iteration prints a summary to the console showing rewards,
losses, and termination statistics.


Expected Results
^^^^^^^^^^^^^^^^

After 15,000 iterations (~4 hours on a single GPU with 512 environments), the Kuka
Allegro hand should reliably grasp and lift the cuboid to target positions.

.. note::

   Training performance depends on hardware, random seed, and physics configuration.
   Newton training may be slower than PhysX due to the more accurate contact solver.
   For best results, use a powerful GPU (e.g., RTX 4090, A100, L40).
