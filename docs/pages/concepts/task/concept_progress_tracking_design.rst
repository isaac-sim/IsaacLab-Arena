Predicates and Subtask Progress Tracking
========================================

A task's success termination tells you if the task was completed. However, this doesn't give
any insight into intermidate goals or stages of the task.
Subtask progress tracking allows for finer-grained tracking of a task's progress.

Arena subtask tracking represents intermediate milestones as **predicates** and organizes them into scored
``ProgressObjective`` objects. The environment builder discovers these objectives from the task,
evaluates them during every environment step, resets them per environment, and exposes their state
and transition history for evaluation and visualization.

Progress tracking does not replace task success and does not terminate an episode. A policy can
receive partial progress even when the task ultimately fails.


Predicates
----------

A predicate is a callable that receives the manager-based environment and returns one Boolean per
parallel environment. For example:

.. code-block:: python

   def object_grasped(env) -> torch.Tensor:
       # Return value shape: (env.num_envs,)
       ...

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

For an order-independent composite task, every child's progress objectives are active. For a
``SequentialTaskBase``, Arena gates each objective per environment using that environment's current
subtask index. A later child's predicates cannot advance before that child becomes active, even if
their physical conditions already happen to be true.

This progress gating follows composite-task ordering, but remains separate from the composite
task's success state. See :doc:`concept_composite_tasks_design` for composition and success
semantics.


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
