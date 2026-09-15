Predicates and Subtask Progress Tracking
========================================

Arena defines task success through ``ProgressObjective`` objects. These objectives organize
Boolean predicates into required milestones, such as settling, lifting, and placing an object.
Their scores also describe partial progress when an episode ends before the task is complete.

Every task returns a ``TaskTerminationCfg`` from ``get_termination_cfg()``. This configuration
declares its ``success`` objectives, named ``failures``, and ``timeout_s`` in one place. The
environment builder creates one success termination that advances the objectives and reports
success when all required objectives are complete.

``NoTask`` declares no success objectives, so environments used for inspection have no success
termination. A task with one success condition uses a single objective with ``predicate_sequences=[predicate]``.


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
* ``object_on_destination`` — destination-footprint, upward-support, and velocity checks for a placement goal.

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

   def object_inside_x_bounds(env, object_name: str, min_x: float, max_x: float) -> torch.Tensor:
       object_x_e = env.arena_world.get_pose_e(object_name)[:, 0]
       return (object_x_e >= min_x) & (object_x_e <= max_x)

The arguments after ``env`` are configured when the predicate is added to a progress objective
(shown in the next section).


Defining a progress objective
-----------------------------

Put the required milestones in ``TaskTerminationCfg.success``. A ``ProgressObjective`` accepts
either ``predicate_sequences`` or ``children``. For ``predicate_sequences``, a list defines one
ordered sequence and a dictionary defines named independent sequences. Use ``children`` to
compose objectives instead; do not supply both arguments.

A sequence is an explicit list, even when it contains only one predicate. The tracker evaluates
its active predicate, ignores later predicates until their turn, and advances by at most one
position per environment step. To weight the predicates, use ``predicate_sequences=[(settled, 1.0), (placed, 3.0)]``;
the tracker normalizes those weights within the sequence.

For example, the built-in pick-and-place task tracks a single progress objective with
a single three-predicate chain: settle, lift, then place.
All three stages must complete in order for the task to succeed. Starting with the object already
on its destination does not complete the lift stage.

.. code-block:: python

   from functools import partial

   from isaaclab.envs import mdp
   from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

   from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
   from isaaclab_arena.tasks.predicates.object_settling import objects_settled
   from isaaclab_arena.tasks.predicates.spatial import object_is_above_height, object_on_destination
   from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

   def get_termination_cfg(self) -> TaskTerminationCfg:
       return TaskTerminationCfg(
           success=[
               ProgressObjective(
                   name="pick_and_place",
                   predicate_sequences=[
                       partial(objects_settled, object_names=[self.pick_up_object.name]),
                       partial(
                           object_is_above_height,
                           object_name=self.pick_up_object.name,
                           use_settled_state=True,
                       ),
                       partial(
                           object_on_destination,
                           object_cfg=SceneEntityCfg(self.pick_up_object.name),
                           destination_cfg=SceneEntityCfg(self.destination_location.name),
                           contact_sensor_cfg=SceneEntityCfg(self.contact_sensor_name),
                           force_threshold=self.force_threshold,
                           velocity_threshold=self.velocity_threshold,
                           support_cone_half_angle_rad=self.support_cone_half_angle_rad,
                       ),
                   ],
               ),
           ],
           failures={
               "object_dropped": TerminationTermCfg(
                   func=mdp.root_height_below_minimum,
                   params={
                       "minimum_height": self.background_scene.object_min_z,
                       "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
                   },
               ),
           },
           timeout_s=self.episode_length_s,
       )

.. note::

    The progress tracker calls each predicate with only ``env``. When a predicate accepts additional
    arguments, use ``functools.partial`` in the progress objective to bind their task-specific values.
    Predicates must be callable directly. The tracker does not initialize manager-term classes or
    resolve scene selections inside bound arguments; predicates that need resolved joint or body
    indices must arrange that preparation explicitly.

For named independent sequences, pass a dictionary to ``predicate_sequences``. Each value is an
explicit predicate list, optionally paired with score weights. Predicates within each sequence
must hold in order, while the named sequences advance independently. A single predicate must
also be wrapped in a list:

.. code-block:: python

   objective = ProgressObjective(
       name="pack_objects",
       predicate_sequences={
           "can": [can_lifted, can_placed],
           "bottle": [bottle_lifted, bottle_placed],
           "box": [box_lifted, box_placed],
       },
       logical="choose",
       K=2,
   )

``logical`` and ``K`` control how completed predicate sequences make the objective complete:

* ``all`` — every sequence must complete. This is the default.
* ``any`` — one sequence must complete.
* ``choose`` — at least ``K`` sequences must complete.

These rules apply to both input forms; a single list counts as one sequence. The type aliases
are ``PredicateSequence`` for one list and ``PredicateSequences`` for a dictionary of named lists.
Progress reports retain their existing group identifiers and fields.

Completed stages are remembered until the environment resets. Separate chains therefore describe
milestones that may complete at different times. If several conditions must hold simultaneously,
combine them into one predicate. For example, checking that all gears are seated together requires
one combined condition; remembering each gear's earlier placement would allow a gear to be removed
before the task completes.


Evaluation and reset lifecycle
------------------------------

``TaskSuccessTerm`` creates and owns ``ProgressTracker`` and connects it to Isaac Lab's ``TerminationManager``:

#. Physics advances and ``TerminationManager`` evaluates ``TaskSuccessTerm``.
#. ``TaskSuccessTerm`` updates ``ProgressTracker`` and returns task completion for that same step.
#. ``ProgressTrackingRecorder`` publishes the resulting state and events to ``env.extras``.
   Reading or recording these results does not evaluate predicates again.
#. Before a completed environment resets, the episode recorder records its final progress.
   ``TerminationManager`` then calls ``TaskSuccessTerm.reset()``, which resets ``ProgressTracker``
   and recorded initial rest poses for the selected environments.

The builder installs this term automatically. It inherits from ``ManagerTermBase`` so Isaac Lab
calls its ``reset()`` when an episode resets. A plain success function could read completion,
but the tracker would need its updates and resets connected elsewhere.

The tracker stores progress state; the root success term manages its updates and resets.
Individual predicates remain ordinary callables;
they do not each need a ``ManagerTermBase`` adapter. Progress tracking needs no separate reset
event or updating recorder. If progress reporting is disabled, success evaluation and resets still work.
Inspect the cached success result through ``env.unwrapped.termination_manager.get_term("success")``.

Consecutive-step predicates remain follow-up work. A sequence orders milestones; it does not
require any predicate to remain true for multiple steps. No ``ForSteps`` API is implemented yet.


Subtask progress tracking in composite and sequential tasks
-----------------------------------------------------------

``CompositeTaskBase`` combines child objectives under one root objective named ``task``.
It namespaces child objectives as ``subtask_<index>/<objective_name>``. Standalone tasks retain
their original objective names, such as ``pick_and_place``. Nested tasks preserve their composition
and ordering inside the same progress tracker.

For an order-independent composite task, every child's progress objectives are active. For a
``SequentialTaskBase``, Arena activates each child only after the preceding child completes in that
environment. The next child starts on the following environment step. A later child's predicates
cannot advance before that child becomes active, even if their physical conditions already happen
to be true.

The composed objective determines both task success and reported progress. Completed child
milestones remain recorded. ``desired_subtask_success_state`` can additionally require selected
children's final conditions to hold, or not hold, when the composed task finishes. See
:doc:`concept_composite_tasks_design` for composition and success semantics.

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

   objective = state.progress_objectives["pick_and_place"]
   print(objective.score, objective.is_complete)
   print(objective.active_predicates)

   for event in progress["events"][env_id]:
       print(event.step, event.progress_objective, event.group, event.predicate_name)

Each ``ProgressObjectiveState`` reports its score, completion state, completed and total group
counts, and the currently active predicate in each group. Each ``PredicateEvent`` records when a
predicate advanced, which objective and group it belongs to, and how much score it contributed.
State and event history are isolated per parallel environment. Reset clears only the selected
environments' tracker state. The published ``extras`` snapshot describes the last evaluated step;
after automatic reset it can still show the episode that just finished until the next step is
published.

Arena's episode recorder also serializes the final progress state and predicate events into the
episode's JSONL record when an output path is configured. Tasks without progress objectives have
no success termination or progress-tracking configuration and produce no progress fields.

For example, one entry of the JSONL record may look like:

.. code-block:: json

   {
     "progress": {
       "overall_score": 0.67,
       "all_complete": false,
       "objectives": {
         "pick_and_place": {
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
           "objective": "pick_and_place",
           "group": "default_group",
           "predicate_index": 0,
           "predicate_name": "objects_settled",
           "score_delta": 0.33
         },
         {
           "step": 18,
           "objective": "pick_and_place",
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
