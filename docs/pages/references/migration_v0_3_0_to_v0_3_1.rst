Migrate from v0.3.0 to v0.3.1
=============================

Arena v0.3.1 upgrades the simulation stack and changes several task, physics, placement, and
runtime-scene interfaces. This guide covers changes that require action in repositories that
import Arena or maintain custom Arena environments.

Quick access
------------

Use this table to jump to the parts that apply to your project.

.. list-table::
   :header-rows: 1

   * - If your v0.3.0 project...
     - Go to
   * - Selects or configures PhysX or Newton
     - :ref:`Physics backend selection <migration-v0-3-1-physics>`
   * - Defines an environment graph or configures relation placement in Python
     - :ref:`Environment graphs and placement defaults <migration-v0-3-1-yaml>`
   * - Defines task success, failure, timeout, or reads progress recordings
     - :ref:`Task completion and progress <migration-v0-3-1-task-completion>`
   * - Uses ``SequentialTaskBase`` or temporal predicate wrappers
     - :ref:`Composite and temporal predicates <migration-v0-3-1-predicates>`
   * - Reads object state in a custom predicate, reward, or metric
     - :ref:`Runtime scene queries <migration-v0-3-1-world>`
   * - Imports ``ObjectType`` or helpers from ``isaaclab_arena.utils.usd``
     - :ref:`Import paths <migration-v0-3-1-imports>`
   * - Depends on the exact number of evaluation episodes
     - :ref:`Exact episode limits <migration-v0-3-1-episodes>`

.. _migration-v0-3-1-physics:

Physics backend selection
-------------------------

The physics backend is now resolved before the environment is composed. Selection follows this
precedence:

#. ``ArenaEnvBuilderCfg.presets`` or the runner's ``--presets`` option.
#. ``IsaacLabArenaEnvironment.default_physics_backend``.
#. ``PhysicsBackend.PHYSX`` when neither setting is provided.

Set a Python environment's default explicitly when it uses Newton:

.. code-block:: python

   from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
   from isaaclab_arena.utils.physics_backend import PhysicsBackend

   environment = IsaacLabArenaEnvironment(
       # ...
       default_physics_backend=PhysicsBackend.NEWTON,
   )

In v0.3.0, an ``env_cfg_callback`` could select a backend by replacing
``env_cfg.sim.physics``. In v0.3.1, the builder materializes the selected ``PhysxCfg`` or
``NewtonCfg`` first, configures the embodiment for that backend, and then invokes the callback.
Use the callback to tune fields on the selected configuration, but do not replace it with the
other backend type. Code that expected ``env_cfg.sim.physics`` to remain ``None`` when no preset
was supplied must now handle the explicit ``PhysxCfg``. The builder also enables
``scene.replicate_physics`` for Newton.

Environment graphs may now select their default backend and apply validated Isaac Lab
configuration overrides:

.. code-block:: yaml

   default_physics_backend: newton
   env_cfg_override:
     sim:
       dt: 0.01
       physics:
         num_substeps: 4

``env_cfg_override`` is applied after backend selection. It may tune the selected physics
configuration, but it cannot replace ``sim.physics``, override ``class_type``, use interpolation
such as ``${...}``, or construct arbitrary classes.

.. _migration-v0-3-1-yaml:

Environment graphs and placement defaults
-----------------------------------------

Move placement validator options from ``placement_validators`` into ``placer_params``. The old
key is rejected:

.. code-block:: yaml

   # v0.3.0
   placement_validators:
     enabled_checks: [no_overlap, on_relation, ik_reachable]
     required_checks: [no_overlap, on_relation]

   # v0.3.1
   placer_params:
     enabled_checks: [no_overlap, on_relation, ik_reachable]
     required_checks: [no_overlap, on_relation]

``placer_params`` can also configure the remaining ``ObjectPlacerParams`` fields and nested
``solver_params`` directly.

The default placement initializer changed from the previous anchor-based behavior to ``on_tree``.
This also affects Python environments that do not supply ``placer_params``. To preserve v0.3.0
placement initialization in YAML, set it explicitly:

.. code-block:: yaml

   placer_params:
     initializer_type: anchor

For a Python environment, pass equivalent parameters to ``IsaacLabArenaEnvironment``:

.. code-block:: python

   from isaaclab_arena.relations.initializers.placement_initializer_base import InitializerType
   from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
   from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

   placer_params = ObjectPlacerParams(
       solver_params=RelationSolverParams(verbose=False, save_position_history=False),
       initializer_type=InitializerType.ANCHOR,
   )
   environment = IsaacLabArenaEnvironment(
       # ...
       placer_params=placer_params,
   )

Assets whose constructors accept an initial pose may also use the structured form:

.. code-block:: yaml

   params:
     initial_pose:
       position_xyz: [-1.0, 0.0, 0.0]
       rotation_xyzw: [0.0, 0.0, 0.0, 1.0]

The task ``composition: atomic|parallel|sequential`` syntax is unchanged.

.. _migration-v0-3-1-task-completion:

Task completion and progress
----------------------------

Task success, failure, and timeout are now described together by ``TaskTerminationCfg``.
``TaskBase.get_progress_objectives()`` is removed, and ``get_termination_cfg()`` no longer accepts
a task-local Isaac Lab ``TerminationsCfg``. Make it return:

* ``timeout_s`` for the episode timeout,
* ``success`` as a list of ``CompletionCriteria``, and
* ``failures`` as a mapping of failure names to Isaac Lab ``TerminationTermCfg`` objects.

Move any v0.3.0 ``success=TerminationTermCfg(...)`` into ``CompletionCriteria``; the same
``TerminationTermCfg`` can go directly in ``predicate_sequence``. Stateful class-based predicates
inside a completion criterion are not reset through Isaac Lab's termination manager, so move
per-episode state into Arena's progress lifecycle or use a stateless predicate. The builder also
stops merging termination configurations from the scene and embodiment. Move any such custom terms
into the task's ``TaskTerminationCfg``.

Move the criteria previously returned by ``get_progress_objectives()`` into ``success``:

.. code-block:: python

   # v0.3.0
   def get_progress_objectives(self):
       return [
           ProgressObjective(
               name="pick_and_place",
               predicate_groups=[object_grasped, object_placed],
           )
       ]

.. code-block:: python

   # v0.3.1
   def get_termination_cfg(self):
       return TaskTerminationCfg(
           timeout_s=self.episode_length_s,
           success=[
               CompletionCriteria(
                   name="pick_and_place",
                   predicate_sequence=[object_grasped, object_placed],
               )
           ],
       )

Use ``predicate_sequence`` for one ordered sequence. Use ``predicate_sequences`` for independently
named sequences and ``CriteriaCompletionMode`` to select ``ALL``, ``ANY``, or ``CHOOSE`` behavior.
This enum replaces ``ProgressObjectiveCompletionMode``. Predicate groups are no longer part of
the public configuration.

The progress recording schema changes with this model. Consumers of evaluation JSONL files must
rename ``objectives``/``objective``/``group`` to
``criteria_by_name``/``criteria_name``/``sequence_name``,
``completed_groups``/``total_groups`` to ``completed_sequences``/``total_sequences``, and the
implicit ``default_group`` name to ``default_sequence``. Old recordings are not converted
automatically; regenerate them or update them before using v0.3.1 reporting tools.

.. _migration-v0-3-1-predicates:

Composite and temporal predicates
---------------------------------

Replace ``SequentialTaskBase`` with ``CompositeTaskBase`` and request ordered execution explicitly:

.. code-block:: python

   # v0.3.0
   task = SequentialTaskBase(subtasks=[pick_task, place_task])

.. code-block:: python

   # v0.3.1
   task = CompositeTaskBase(
       subtasks=[pick_task, place_task],
       subtasks_are_sequential=True,
   )

Pass a flat list of atomic subtasks. Nested ``CompositeTaskBase`` instances are not supported.

Define a predicate as a pure current-state check, then wrap its configuration when
consecutive-step behavior is needed:

.. code-block:: python

   from functools import partial

   from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
   from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

   objects_are_resting = partial(objects_below_velocity_thresholds, object_names=["cube"])

   CompletionCriteria(
       name="settled",
       predicate_sequence=[
           TrueForConsecutiveStepsCfg(
               predicate=objects_are_resting,
               required_steps=10,
           )
       ],
   )

The completion runner owns and resets the streak counter. The wrapped predicate should only report
the current step's Boolean result.

.. _migration-v0-3-1-world:

Runtime scene queries
---------------------

Custom predicates, rewards, and metrics should accept ``IsaacLabArenaManagerBasedRLEnv`` and use
its new read-only ``env.arena_world`` facade for Arena object state:

.. code-block:: python

   T_W_O = env.arena_world.get_pose_w(object_name)
   position_e = env.arena_world.get_pose_e(object_name)[:, :3]
   linear_velocity_w = env.arena_world.get_root_linear_velocity_w(object_name)
   bounds_w = env.arena_world.get_aabb_w(object_name)

The modules ``tasks.predicates.live_scene_geometry``, ``tasks.predicates.predicate_utils``, and
``tasks.predicates.object_on_destination_term`` and the
``tasks.observations.observations.object_position_in_world_frame`` helper were removed. Replace
them with ``ArenaWorld`` queries or current predicates. Direct ``env.scene[...]`` access is still
available when custom code specifically needs an Isaac Lab sensor or native asset handle, but
``ArenaWorld`` is the compatible path for heterogeneous and deformable Arena objects.

``ObjectBase.get_object_pose()`` is also removed. Query the object's scene key through
``env.arena_world.get_pose_e()`` or ``get_pose_w()`` instead.

Built-in state predicates now return values for all environments. Remove the ``env_id`` argument
from calls to ``objects_settled``, ``object_is_above_height``, and ``object_moving``.

The ``object_on_destination`` predicate now requires ``destination_cfg``, ``force_threshold``, and
``velocity_threshold``. It also requires the object's bounds center to be over the destination and
the rigid contact force to fall within the upward support cone, so results may change after the
call is updated:

.. code-block:: python

   TerminationTermCfg(
       func=object_on_destination,
       params={
           "object_cfg": SceneEntityCfg("object"),
           "destination_cfg": SceneEntityCfg("destination"),
           "contact_sensor_cfg": SceneEntityCfg("object_contact_sensor"),
           "force_threshold": 0.1,
           "velocity_threshold": 0.003,
       },
   )

``objects_on_destinations`` was removed. Build one ``TerminationTermCfg`` per object and combine
them with ``check_success(..., mode=SuccessMode.ALL)`` inside one completion predicate; do not put
them in a ``predicate_sequence``, which would make the checks ordered.

``PickAndPlaceTask.support_cone_half_angle_deg`` became
``support_cone_half_angle_rad``. Convert the value as well as the name, for example from ``45.0``
to ``math.pi / 4``. Apply the same rename and unit conversion to ``params`` in environment graph
YAML.

.. _migration-v0-3-1-imports:

Import paths
------------

The ``isaaclab_arena.utils.usd`` package does not re-export its helper functions. Import them from
``isaaclab_arena.utils.usd.helpers`` or ``isaaclab_arena.utils.usd.rigid_bodies``. Import
``ObjectType`` from ``isaaclab_arena.assets.object_type``.

.. _migration-v0-3-1-episodes:

Exact episode limits
--------------------

``rollout_limit.num_episodes`` now means exactly that many episodes are started and completed,
including when the value is not divisible by ``num_envs``. Result counts may therefore be lower
than in v0.3.0 runs that overshot the requested limit.

The Experiment and Policy Runners configure this automatically. Direct environment users can
request the same behavior, but must do so before the initial reset:

.. code-block:: python

   env.unwrapped.configure_episode_limit(7)
   observations, info = env.reset()
