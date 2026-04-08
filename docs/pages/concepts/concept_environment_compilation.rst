Environment Compilation
=======================

Environment compilation is the step that turns the three independent components —
scene, embodiment, and task — into a runnable Isaac Lab environment.
``ArenaEnvBuilder`` does this by collecting the partial configuration each
component contributes and merging them into a single
``ManagerBasedRLEnvCfg``.

.. figure:: ../../images/arena_env_builder.png
   :width: 100%
   :alt: ArenaEnvBuilder merges Scene, Embodiment, and Task into a ManagerBasedRLEnv
   :align: center

   ``ArenaEnvBuilder`` merges the Scene, Embodiment, and Task into a runnable ``ManagerBasedRLEnv``.

.. code-block:: python

   environment = IsaacLabArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,
       task=task,
   )

   env_builder = ArenaEnvBuilder(environment, args_cli)
   env = env_builder.make_registered()

How it works
------------

Each component exposes a set of ``get_*_cfg()`` methods that return its
contribution to each Isaac Lab manager:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 25

   * - Manager
     - Scene
     - Embodiment
     - Task
   * - Scene
     - assets, lights
     - robot, sensors
     - task-specific assets
   * - Observations
     -
     - proprioception, cameras
     - goal observations
   * - Actions
     -
     - control interface
     -
   * - Events (resets)
     - object placement
     - robot reset
     - task reset
   * - Terminations
     -
     -
     - success, failure
   * - Rewards
     -
     -
     - dense rewards (RL)

``ArenaEnvBuilder.compose_manager_cfg()`` merges all of these into one config.
It also wires up the metrics recorder manager from the task's ``get_metrics()``,
and optionally solves spatial relations between objects (``--solve_relations``).

The compiled config is then registered with the gym registry under the
environment's name, and ``gym.make()`` returns the running environment.

Mimic mode
----------

Passing ``--mimic`` at the command line compiles a
``ManagerBasedRLMimicEnv`` instead of a standard ``ManagerBasedRLEnv``.
The mimic environment is used for demonstration generation and includes
subtask configurations from the task. Metrics and recorders are excluded
in mimic mode.

.. code-block:: bash

   python my_script.py --mimic ...
