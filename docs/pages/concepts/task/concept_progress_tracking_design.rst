Predicates and Subtask Progress Tracking
========================================

A task's success termination tells you if the task was completed. However, this doesn't give
any insight into intermediate goals or stages of the task.
Subtask progress tracking allows for finer-grained tracking of a task's progress.

Arena subtask tracking represents intermediate milestones as **predicates** and organizes them into scored
``ProgressObjective`` objects. The environment builder discovers these objectives from the task,
evaluates them during every environment step, resets them per environment, and exposes their state
and transition history for evaluation and visualization.

Progress tracking does not replace task success and does not terminate an episode. A policy can
receive partial progress even when the task ultimately fails.


Predicates
----------

A predicate represents a boolean condition in a task, such as an object settling,
being lifted, or reaching its destination. In Arena, a predicate is a callable that receives
the manager-based environment (and optionally additional configuration arguments) and returns one Boolean per parallel environment.

Included predicates
~~~~~~~~~~~~~~~~~~~

Arena comes with an existing collection of predicates under ``isaaclab_arena.tasks.predicates``, including:

* ``objects_settled`` — all selected objects are below linear and angular velocity thresholds.
* ``object_is_above_height`` — an object is above a fixed height or its recorded resting height.
* ``object_moving`` — an object exceeds a linear velocity threshold.
* ``objects_in_proximity`` — two objects are within configured axis-aligned distances.
* ``object_on_destination`` and ``objects_on_destinations`` — contact and velocity checks for
  placement goals.

.. note::

    ``objects_settled`` records each object's first resting pose. Later predicates can use that
    environment-specific pose as a reference, which is more robust than assuming every object starts at
    the same world height. Arena clears the recorded poses for the environments being reset.


Defining a custom predicate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom predicate when Arena's included predicates do not express the condition you need.
A custom predicate must:

* Accept ``env`` as its first argument.
* Evaluate all parallel environments in one call.
* Return a Boolean tensor with shape ``(env.num_envs,)``.

A predicate may accept any task-specific arguments it needs after ``env``. For example:

.. code-block:: python

   import torch

   from isaaclab_arena.tasks.predicates.predicate_utils import get_root_pos_w

   def object_inside_x_bounds(env, object_name: str, min_x: float, max_x: float) -> torch.Tensor:
       object_x = get_root_pos_w(env, object_name)[:, 0]
       return (object_x >= min_x) & (object_x <= max_x)

The arguments after ``env`` are configured when the predicate is added to a progress objective
(shown in the next section).


Defining a progress objective
-----------------------------

Subtask tracking can be added to tasks via opt in by overriding ``TaskBase.get_progress_objectives()``.
A ``ProgressObjective`` is a named collection of predicates that the task wishes to track.
A ``predicate_groups`` contains ordered predicate chains. For each chain, the tracker evaluates only the currently active
predicate, ignores later predicates until their turn, and advances by at most one predicate per
group and environment step.

For example, the built-in pick-and-place task tracks a single progress objective with
a single three-predicate chain: settle, lift, then place.

.. code-block:: python

   from functools import partial

   from isaaclab.managers import SceneEntityCfg

   from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
   from isaaclab_arena.tasks.predicates.object_settling import objects_settled
   from isaaclab_arena.tasks.predicates.spatial import object_is_above_height, object_on_destination

   def get_progress_objectives(self) -> list[ProgressObjective]:
       return [
           ProgressObjective(
               name="pick_and_place",
               predicate_groups=[
                   partial(objects_settled, object_names=[self.pick_up_object.name]),
                   partial(
                       object_is_above_height,
                       object_name=self.pick_up_object.name,
                       use_settled_state=True,
                   ),
                   partial(
                       object_on_destination,
                       object_cfg=SceneEntityCfg(self.pick_up_object.name),
                       contact_sensor_cfg=SceneEntityCfg(self.contact_sensor_name),
                   ),
               ],
           )
       ]

.. note::

    The progress tracker calls each predicate with only ``env``. When a predicate accepts additional
    arguments, use ``functools.partial`` in the progress objective to bind their task-specific values.

Use a dictionary to specify ``predicate_groups`` when a progress objective has multiple independent predicate chains. Predicates
within each group remain sequential while groups advance independently:

.. code-block:: python

   objective = ProgressObjective(
       name="pack_objects",
       predicate_groups={
           "can": [can_lifted, can_placed],
           "bottle": [bottle_lifted, bottle_placed],
           "box": [box_lifted, box_placed],
       },
       logical="choose",
       K=2,
   )

``logical`` controls how completed groups make the objective complete:

* ``all`` — every group must complete. This is the default.
* ``any`` — one group must complete.
* ``choose`` — at least ``K`` groups must complete.


Subtask progress tracking in composite and sequential tasks
-----------------------------------------------------------

``CompositeTaskBase`` collects the progress objectives from every child and namespaces their names
as ``subtask_<index>/<objective_name>``. It also records the child index on each objective.
Standalone tasks retain their original objective names, such as ``pick_and_place``. The
``subtask_<index>/`` prefix is added only when the task is part of a composite task.

For an order-independent composite task, every child's progress objectives are active. For a
``SequentialTaskBase``, Arena gates each objective per environment using that environment's current
subtask index. A later child's predicates cannot advance before that child becomes active, even if
their physical conditions already happen to be true.

This progress gating follows composite-task ordering, but remains separate from the composite
task's success state. See :doc:`concept_composite_tasks_design` for composition and success
semantics.

.. figure:: ../../../images/composite_vs_sequential_progress_tracking.png
   :width: 100%
   :alt: Comparison of predicate tracking activation in composite and sequential tasks
   :align: center

   Composite tasks activate tracking on all subtasks' predicates together, while sequential tasks activate
   tracking on each subtask's predicates only after the preceding subtask succeeds.


Reading subtask progress tracking at runtime
--------------------------------------------

When a task provides progress objectives, Arena will track and record the progress of the task according to the
supplied progress objectives. The current per-environment state and the episode's accumulated predicate
transitions are available through ``env.extras``:

.. code-block:: python

   progress = env.unwrapped.extras["progress_tracking"]

   state = progress["states"][env_id]
   print(state.overall_score, state.all_complete)

   objective = state.progress_objectives["subtask_0/pick_and_place"]
   print(objective.score, objective.is_complete)
   print(objective.active_predicates)

   for event in progress["events"][env_id]:
       print(event.step, event.progress_objective, event.group, event.predicate_name)

Each ``ProgressObjectiveState`` reports its score, completion state, completed and total group
counts, and the currently active predicate in each group. Each ``PredicateEvent`` records when a
predicate advanced, which objective and group it belongs to, and how much score it contributed.
State and event history are isolated per parallel environment and cleared only for environments
that reset.

Arena's episode recorder also serializes the final progress state and predicate events into the
episode's JSONL record when an output path is configured. Tasks without progress objectives have
no progress-tracking configuration and produce no progress fields.

For example, one entry of the JSONL record may look like:

.. code-block:: json

   {
     "progress": {
       "overall_score": 0.67,
       "all_complete": false,
       "objectives": {
         "subtask_0/pick_and_place": {
           "score": 0.67,
           "is_complete": false,
           "completed_groups": 0,
           "total_groups": 1,
           "active_predicates": {
             "default_group": "object_on_destination"
           }
         }
       },
       "events": [
         {
           "step": 4,
           "objective": "subtask_0/pick_and_place",
           "group": "default_group",
           "predicate_index": 0,
           "predicate_name": "objects_settled",
           "score_delta": 0.33
         },
         {
           "step": 18,
           "objective": "subtask_0/pick_and_place",
           "group": "default_group",
           "predicate_index": 1,
           "predicate_name": "object_is_above_height(object_name='can', use_settled_state=True)",
           "score_delta": 0.33
         }
       ]
     }
   }

In this example, the object has settled and then been lifted, completing two of the three predicates
and producing a progress score of ``0.67``. The objective is not complete however because the final predicate
(``object_on_destination``) has not been satisfied. The events record when each completed predicate advanced
the task and how much it contributed to the score. Here, there have been two events recorded so far, one for when
the ``objects_settled`` predicate was satisfied and one for when the ``object_is_above_height`` predicate was satisfied.
