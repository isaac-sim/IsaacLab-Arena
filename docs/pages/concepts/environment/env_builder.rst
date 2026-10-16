Environment Builder
===================

Environment compilation is the step that turns the three independent components —
scene, embodiment, and task — into a runnable Isaac Lab environment.
``ArenaEnvBuilder`` does this by collecting the partial configuration each
component contributes and merging them into a single
``ManagerBasedRLEnvCfg``.

.. figure:: ../../../images/arena_env_builder.png
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

   env_builder = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg())
   env = env_builder.make_registered()

How it works
------------

Each component (Scene, Embodiment, Task) exposes a set of ``get_*_cfg()`` methods that return its
contribution to each Isaac Lab manager. The typical contributions of each component
to each manager are tabulated below:

+-------------+------------------------------------------------------------------------+
| Isaac Lab   | Isaac Lab - Arena Component                                            |
+ Manager     +-----------------------+-------------------------+----------------------+
|             | Scene                 | Embodiment              | Task                 |
+=============+=======================+=========================+======================+
| Scene       | assets, lights        | robot, sensors          | task-specific assets |
+-------------+-----------------------+-------------------------+----------------------+
| Observations|                       | proprioception, cameras | goal observations    |
+-------------+-----------------------+-------------------------+----------------------+
| Actions     |                       | control interface       |                      |
+-------------+-----------------------+-------------------------+----------------------+
| Events      | object placement      | robot reset             | task reset           |
| (resets)    |                       |                         |                      |
+-------------+-----------------------+-------------------------+----------------------+
| Terminations|                       |                         | success, failure     |
+-------------+-----------------------+-------------------------+----------------------+
| Rewards     |                       |                         | dense rewards (RL)   |
+-------------+-----------------------+-------------------------+----------------------+
| Recorder    |                       |                         | metrics-required data|
+-------------+-----------------------+-------------------------+----------------------+


``ArenaEnvBuilder.compose_manager_cfg()`` first assembles the partial manager contributions
from each component into a set of complete managers. Then it merges these complete managers
into a single ``ManagerBasedRLEnvCfg``.

By default, the builder also solves spatial relations for placed objects and
supported robot embodiments. Set
``ArenaEnvBuilderCfg(solve_relations=False)`` in Python to disable this step.
In an Experiment Run, set ``environment_builder.solve_relations: false``. The
Environment Runner exposes the equivalent ``--no_solve_relations`` option.

Seeds during compilation
------------------------

Environment compilation and rollout use more than one random stream, so locking
a single global seed is not enough when you need layouts, object-set picks, or
run-time variation draws to be reproducible independently.

What it is
~~~~~~~~~~

``ArenaEnvBuilder`` exposes two seeds. They are independent — locking one does
not fix the other.

.. list-table::
   :header-rows: 1
   :widths: 18 28 12 42

   * - Control
     - Experiment YAML / config
     - Default
     - Locking it reproduces
   * - Environment seed
     - ``environment_builder.seed`` / ``ArenaEnvBuilderCfg.seed``
     - ``42``
     - Simulation RNG after the Isaac Lab env is created: reset noise and
       :doc:`run-time variation <../variations/variations>` draws.
   * - Placement seed
     - ``environment_builder.placement_seed`` /
       ``ArenaEnvBuilderCfg.placement_seed``
     - ``None`` (unlocked)
     - Relation-solver layouts and random
       :doc:`RigidObjectSet <../scene/concept_rigid_object_set>` member
       assignment. With ``None``, placement stays non-reproducible across runs.

There is no variation seed. Run-time variations follow the environment seed;
build-time variations are drawn once at compile time and are not locked by
either seed. See :doc:`../variations/variations` and
:doc:`../object_placement/pooled_placement`.

How to set it
~~~~~~~~~~~~~

Set the seeds you want to lock under ``environment_builder`` in an Experiment
Run (or set the matching field on ``ArenaEnvBuilderCfg`` / ``placer_params`` in
Python):

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/concepts/maple_table_seeds_experiment.yaml
   :language: yaml
   :start-after: SPDX-License-Identifier: Apache-2.0

Run it from the repository root:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/concepts/maple_table_seeds_experiment.yaml

- Set ``seed`` to fix simulation and run-time variation draws.
- Set ``placement_seed`` to fix layouts and random object-set picks.
- Leave ``placement_seed`` unset when placement should vary across runs. To
  unlock it in this example without editing the YAML, append
  ``runs.locked_seeds.environment_builder.placement_seed=null`` to the
  command.

The compiled config is then registered with the gym registry under the
environment's name, and ``gym.make()`` returns the gym environment.

Mimic mode
----------

Passing ``--mimic`` at the command line compiles a
``ManagerBasedRLMimicEnv`` instead of a standard ``ManagerBasedRLEnv``.
The mimic environment is used for demonstration generation and includes
subtask configurations from the task. Metrics and recorders are excluded
in mimic mode.

.. code-block:: bash

   python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --mimic ...

Next Steps
----------

Continue to :doc:`../object_placement/relations` to learn how anchors and spatial
relations describe a placement layout.
