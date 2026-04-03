Policy Training (Isaac Lab)
----------------------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

.. important::

   Training is performed **in Isaac Lab** (not Arena). To use **Newton**
   physics, add ``presets=newton`` to the training command. Without
   ``presets=newton``, the training will use default PhysX backend.
   It's important to use the same physics backend for training and evaluation,
   to avoid the sim-to-sim gap between PhysX and Newton.


Training Command
^^^^^^^^^^^^^^^^

Train the ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` task with Isaac Lab's RSL-RL
training script:

.. code-block:: bash

   # Newton physics (recommended for this example):
   python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
     --num_envs 512 \
     presets=newton presets=cube

``presets=newton`` selects the Newton physics backend and ``presets=cube``
switches the manipulation object to a single-geometry cube (Newton does not
support multi-asset spawning used by the default ``shapes`` preset).

This uses the ``DexsuiteKukaAllegroPPORunnerCfg`` configuration defined in
Isaac Lab, which provides:

- **Actor/Critic**: MLP [512, 256, 128], ELU activation, observation normalization enabled
- **Observation groups**: ``policy`` + ``proprio`` + ``perception`` (all three groups
  concatenated, each with 5-step history)
- **Algorithm**: PPO with adaptive learning rate schedule, starting at ``1e-3``
- **Training**: 15,000 iterations, 32 steps per environment, 512 parallel environments
- **Physics**: Newton (MuJoCo-Warp solver) when ``presets=newton`` is used

Checkpoints are saved every 250 iterations to
``logs/rsl_rl/dexsuite_kuka_allegro/<timestamp>/``.

.. tip::

   Add ``--visualizer newton`` to visualize training with the Newton (MuJoCo) viewer.


Overriding Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Hyperparameters can be overridden with Hydra-style CLI arguments:

.. code-block:: bash

   python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
     --num_envs 512 \
     presets=newton presets=cube \
     agent.max_iterations=20000 agent.save_interval=500 agent.algorithm.learning_rate=0.0005


Monitoring Training
^^^^^^^^^^^^^^^^^^^

Launch Tensorboard to monitor progress:

.. code-block:: bash

   python -m tensorboard.main --logdir logs/rsl_rl

During training, each iteration prints a summary to the console showing rewards,
losses, and termination statistics.


Expected Results
^^^^^^^^^^^^^^^^

After 15,000 iterations (~4 hours on a single GPU with 512 environments), the
Kuka Allegro hand should reliably grasp and lift the cuboid to target positions.

.. note::

   Training performance depends on hardware, random seed, and physics
   configuration. Newton training may be slower than PhysX due to the more
   accurate contact solver. For best results, use a powerful GPU (e.g., RTX
   4090, A100, L40).
