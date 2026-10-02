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
Arena command-line runners expose the equivalent ``--no_solve_relations``
option.

Seeds during compilation
------------------------

Environment compilation and rollout use more than one random stream, so locking
a single global seed is not enough when you need layouts, object-set picks, or
run-time variation draws to be reproducible independently.

What it is
~~~~~~~~~~

``ArenaEnvBuilder`` exposes three seeds. They are independent; locking one does
not fix the others.

.. list-table::
   :header-rows: 1
   :widths: 18 28 12 42

   * - Control
     - CLI / config
     - Default
     - Locking it reproduces
   * - Environment seed
     - ``--seed`` / ``ArenaEnvBuilderCfg.seed``
     - ``42``
     - Simulation RNG after the Isaac Lab env is created: reset noise and
       :doc:`run-time variation <../variations/variations>` draws.
   * - Placement seed
     - ``--placement_seed`` / ``ArenaEnvBuilderCfg.placement_seed``
     - ``None`` (unlocked)
     - Relation-solver layouts and random
       :doc:`RigidObjectSet <../scene/concept_rigid_object_set>` member
       assignment. With ``None``, placement stays non-reproducible across runs.

   * - Variation seed
     - ``--variation_seed`` / ``ArenaEnvBuilderCfg.variation_seed``
     - ``None`` (legacy sampling)
     - Built-in sampler draws keyed by variation path, environment ID, and episode
       index, independently of reset ordering and unrelated variations.

Without an explicit variation seed, run-time variations follow ``--seed`` and
build-time draws retain their existing random stream. Set ``variation_seed`` in
Python or an Experiment Definition's ``environment_builder`` mapping to control
both. ``variation_replay_path`` accepts one Run/rebuild's recorded episode JSONL
and requires a recorded value for every enabled variation and requested episode.
It replays variation values, not robot trajectories or complete physical state.
See :doc:`../variations/variations` and
:doc:`../object_placement/pooled_placement`.

How to set it
~~~~~~~~~~~~~

Pass the seed you want to lock on the runner CLI (or set the matching field on
``ArenaEnvBuilderCfg`` / ``placer_params``):

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --seed 42 \
     --placement_seed 7 \
     --num_steps 100 \
     pick_and_place_maple_table

- Set ``--seed`` to fix simulation and run-time variation draws.
- Set ``--placement_seed`` to fix layouts and random object-set picks.
- Omit ``--placement_seed`` when placement should vary across runs.

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

Shared task state
-----------------

A task that owns instruments or coupled mechanisms may return a ``TaskRuntimeCfg``
from ``get_runtime_cfg()``. Arena constructs the runtime once after scene handles
exist. ``prepare_reset(env_ids)`` releases episode constraints before ordinary reset
and variation events; ``reset(env_ids)`` establishes coherent episode state after
those events. Episode recording happens before either hook modifies terminal state.

``update()`` runs immediately before success/progress evaluation, after physics.
Predicates and observations read its snapshot. Implementations must make repeated
updates at the same control step idempotent and reset only selected environment IDs.
Failure terms run before this success hook, so they must not depend on its new
snapshot. This optional lifecycle currently requires a task with success criteria
and the standard Arena environment; Mimic compilation rejects it explicitly.

``CompositeTaskBase`` forwards a single child's runtime configuration. If several
children require shared state, override ``get_runtime_cfg()`` on the composite
and return one owner for that state; implicit multiple runtimes are rejected.
