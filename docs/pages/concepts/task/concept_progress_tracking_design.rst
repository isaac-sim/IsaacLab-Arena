Predicates and Subtask Progress Tracking
========================================

Arena defines task success through ``CompletionCriteria`` objects. These criteria sets organize
Boolean predicates into required milestones, such as lifting and placing an object.
Their scores also describe partial progress when an episode ends before the task is complete.

Every task returns a ``TaskTerminationCfg`` from ``get_termination_cfg()``. This configuration
declares its ``success`` criteria, named ``failures``, and ``timeout_s`` in one place. The
environment builder creates one success termination that advances the criteria sets and reports
success when all required criteria sets are complete.


Predicates
----------

A predicate represents a boolean condition in a task, such as an object settling,
being lifted, or reaching its destination. An instantaneous predicate is a callable that receives
the manager-based environment and returns one Boolean per parallel environment. ``ObjectLifted``
and consecutive-step requirements also retain episode state. The tracker manages activation and
reset for these checks, with separate state for each configured occurrence.

Included predicates
~~~~~~~~~~~~~~~~~~~

Arena comes with an existing collection of predicates under ``isaaclab_arena.tasks.predicates``, including:

* ``objects_below_velocity_thresholds`` — all selected objects are below linear and angular velocity thresholds.
* ``object_is_above_height`` — an object is above a fixed reference height.
* ``ObjectLifted`` — captures a reference height on first active evaluation and detects a subsequent rise.
* ``object_moving`` — an object exceeds a linear velocity threshold.
* ``objects_in_proximity`` — two objects are within configured axis-aligned distances.
* ``object_on_destination`` — destination-footprint, upward-support, and velocity checks for a placement goal.

.. note::

    Each ``ObjectLifted`` occurrence owns its reference height per environment.
    It captures that height when it first becomes active and keeps it until reset.
    It does not check whether the object is at rest.
    ``PickAndPlaceTask`` first requires five consecutive low-velocity control steps through an
    unscored prerequisite, configurable through ``settling_steps``. The lift predicate then captures
    the height. This prerequisite checks velocity; it does not check contact or support.
    Policy actions continue during settling. Any earlier movement becomes part of the reference;
    the object must rise again after that reference is captured to satisfy the lift predicate.

    The shared ``ObjectInitialRestPoseRecorder`` and ``use_settled_state`` argument have been removed.
    Use a settling prerequisite followed by ``ObjectLifted`` for a measured reference;
    use ``object_is_above_height`` with ``surface_height`` for a fixed reference.
    ``objects_below_velocity_thresholds`` only checks current velocity.


Defining a custom predicate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom predicate when Arena's included predicates do not express the condition you need.
An instantaneous predicate must:

* Accept ``env`` as its first argument.
* Evaluate all parallel environments in one call.
* Return a Boolean tensor with shape ``(env.num_envs,)``.

A predicate may accept any task-specific arguments it needs after ``env``. For example:

.. code-block:: python

   import torch

   def object_inside_x_bounds(env, object_name: str, min_x: float, max_x: float) -> torch.Tensor:
       object_x_e = env.arena_world.get_pose_e(object_name)[:, 0]
       return (object_x_e >= min_x) & (object_x_e <= max_x)

The arguments after ``env`` are configured when the predicate is added to a criteria set
(shown in the next section).


Defining completion criteria
-----------------------------

Use ``prerequisites`` for preparation conditions that must hold together before a criteria set's
predicate sequences begin:

.. code-block:: python

   CompletionCriteria(
       name="pick_and_place",
       prerequisites=[settled],
       predicate_sequence=[lifted, placed],
   )

Prerequisites earn no score or completion events. Readiness is remembered separately per environment
until reset; the first sequence predicate can run in the same update that establishes readiness.
The runner's state exposes ``prerequisites_met`` without evaluating the conditions again.
For sequential subtasks, prerequisites begin when the subtask becomes active.
Policy actions and the episode clock continue while prerequisites are pending.
Ordinary callable prerequisites evaluate the full batch and should be free of state updates;
the runner updates lift references and consecutive-step counters only for waiting environments
in active criteria sets. It clears that state for the environments being reset.

Add ``CompletionCriteria`` entries to ``TaskTerminationCfg.success``. Provide exactly one of
``predicate_sequence`` for a list of predicates or ``predicate_sequences`` for a dictionary of named lists.

``PickAndPlaceTask`` requires the object to be lifted and then placed. A settling prerequisite
runs before the lift predicate captures its reference, without treating settling as a success milestone:

.. code-block:: python

   from functools import partial

   from isaaclab.envs import mdp
   from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

   from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
   from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted
   from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
   from isaaclab_arena.tasks.predicates.spatial import object_on_destination
   from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
   from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

   def get_termination_cfg(self) -> TaskTerminationCfg:
       settled = TrueForConsecutiveStepsCfg(
           predicate=partial(objects_below_velocity_thresholds, object_names=[self.pick_up_object.name]),
           required_steps=self.settling_steps,
       )
       lifted = TerminationTermCfg(func=ObjectLifted, params={"object_name": self.pick_up_object.name})
       return TaskTerminationCfg(
           success=[
               CompletionCriteria(
                   name="pick_and_place",
                   prerequisites=[settled],
                   predicate_sequence=[
                       lifted,
                       TrueForConsecutiveStepsCfg(
                           predicate=partial(
                               object_on_destination,
                               object_cfg=SceneEntityCfg(self.pick_up_object.name),
                               destination_cfg=SceneEntityCfg(self.destination_location.name),
                               contact_sensor_cfg=self.contact_sensor_cfg,
                               force_threshold=self.force_threshold,
                               velocity_threshold=self.velocity_threshold,
                               support_cone_half_angle_rad=self.support_cone_half_angle_rad,
                           ),
                           required_steps=self.placement_consecutive_steps,
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

Configuring a predicate's arguments and requiring it to stay true are separate choices.
Here, ``partial`` supplies the object names; ``CompletionCriteriaRunner`` supplies the environment
when it evaluates the predicate:

.. code-block:: python

   from functools import partial

   from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
   from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

   objects_are_resting = partial(objects_below_velocity_thresholds, object_names=["cube"])

   # Complete this entry when the condition is true for one step.
   predicate_sequence = [objects_are_resting]

   # Or require the same condition to hold for ten consecutive steps.
   predicate_sequence = [
       TrueForConsecutiveStepsCfg(
           predicate=objects_are_resting,
           required_steps=10,
       ),
   ]

``TerminationTermCfg`` is another way to supply the function and its arguments instead of ``partial``:

.. code-block:: python

   from isaaclab.managers import TerminationTermCfg

   objects_are_resting = TerminationTermCfg(
       func=objects_below_velocity_thresholds,
       params={"object_names": ["cube"]},
   )

This configuration can also go directly in ``predicate_sequence`` or inside ``TrueForConsecutiveStepsCfg``.
``CompletionCriteriaRunner`` prepares the configuration, including nested predicates, and evaluates
the resulting runtime; ``TerminationTermCfg`` does not create a separate termination-manager term
here. The runner initializes a configured class with ``(cfg, env)``;
``ManagerTermBase`` inheritance is not required. Ordinary callable classes evaluate the full batch.
The tracker explicitly manages active-environment masks and episode resets for ``ObjectLifted``
and consecutive-step runtimes. Inheriting ``ManagerTermBase`` or defining a ``reset()`` method alone
does not enable that behavior for another class. Other predicates that retain episode state must
clear it through their own reset path.

``PickAndPlaceTask`` defaults to ``placement_consecutive_steps=1``. Set it to a larger positive
integer, such as ``10``, to require placement, support, and low speed to hold together for that
many consecutive control steps.

Use a dictionary to track several sequences independently. This example requires any two
objects to be lifted and placed. Each entry, such as ``can_lifted``, is a configured callable:

.. code-block:: python

   criteria = CompletionCriteria(
       name="pack_objects",
       predicate_sequences={
           "can": [can_lifted, can_placed],
           "bottle": [bottle_lifted, bottle_placed],
           "box": [box_lifted, box_placed],
       },
       logical="choose",
       K=2,
   )

``logical`` and ``K`` control how completed predicate sequences make the criteria set complete:

* ``all`` — every sequence must complete. This is the default.
* ``any`` — one sequence must complete.
* ``choose`` — at least ``K`` sequences must complete.

Completed stages are remembered until the environment resets. Separate sequences therefore describe
milestones that may complete at different times. If several conditions must hold simultaneously,
combine them into one predicate. For example, checking that all gears are seated together requires
one combined condition; remembering each gear's earlier placement would allow a gear to be removed
before the task completes.


Conditions that must remain true
--------------------------------

Use ``TrueForConsecutiveStepsCfg`` around an instantaneous predicate, a configured ``ObjectLifted``,
or another consecutive-step requirement:

.. code-block:: python

   placement_held = TrueForConsecutiveStepsCfg(
       predicate=placed_and_stable,
       required_steps=10,
   )

``placed_and_stable`` returns one Boolean per environment; it does not maintain a counter.
``CompletionCriteriaRunner`` prepares each ``TrueForConsecutiveStepsCfg`` occurrence recursively,
creating a fresh ``_TrueForConsecutiveSteps`` runtime and its child. The runtime owns its counter
and forwards activation and resets to a child that is ``ObjectLifted`` or another consecutive-step
runtime. Ordinary callable children are evaluated without active masks or automatic resets.
For active environments, a true child result adds one to the counter and false clears the streak.
``CompletionCriteriaRunner`` owns its prepared prerequisite and sequence entries. It selects active
environments for lift and consecutive-step checks and resets them through
``TaskSuccessTerm`` / ``ProgressTracker``. Each consecutive-step runtime then resets its supported child.

For example, this configuration captures a lift reference when active and requires the object to
remain above that reference for three consecutive steps:

.. code-block:: python

   held_lift = TrueForConsecutiveStepsCfg(
       predicate=TerminationTermCfg(func=ObjectLifted, params={"object_name": "cube"}),
       required_steps=3,
   )

Reusing ``held_lift`` in another sequence position creates a separate counter and lift reference.
The tracker shares cached results within one control-step update, including nested evaluations and
final-condition checks. An ordinary callable is evaluated once by identity for the full batch.
Each lift or consecutive-step occurrence updates only the requested environments that have not yet
been evaluated during that step. Reading a cached result does not activate another environment or
advance state again.

``TaskSuccessTerm`` advances ``ProgressTracker`` once per control step. Reporting and other consumers
read ``is_complete()``, ``get_state()``, or ``get_events()`` without advancing progress.
Direct callers of ``ProgressTracker.step()`` must also call it exactly once per control step.
Temporal requirements need a ``step_index`` per environment, such as ``env.episode_length_buf``.
After the first update, repeated, skipped, or backwards indices raise an assertion before any
predicates are evaluated or counters change. ``ProgressTracker.reset()`` clears the stored index
for each restarting environment.

The examples below show three different requirements. ``object_still(env)`` and ``gripper_slow(env)``
are configured instantaneous checks that each return one Boolean per environment.
Step numbers start when the criteria become active.

One condition after another
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put both requirements in one ``predicate_sequence`` to count them in order:

.. code-block:: python

   criteria = CompletionCriteria(
       name="object_then_gripper",
       predicate_sequence=[
           TrueForConsecutiveStepsCfg(object_still, required_steps=10),
           TrueForConsecutiveStepsCfg(gripper_slow, required_steps=10),
       ],
   )

``CompletionCriteriaRunner`` first waits for ten consecutive steps with the object still.
On the following step, it starts counting the gripper's ten steps. The earliest completion is
step 20. The object may move again after its requirement completes; that completion is remembered.

Independent conditions, both completed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use separate named ``predicate_sequences`` to start both counters together:

.. code-block:: python

   criteria = CompletionCriteria(
       name="object_and_gripper_ready",
       predicate_sequences={
           "object": [TrueForConsecutiveStepsCfg(object_still, required_steps=10)],
           "gripper": [TrueForConsecutiveStepsCfg(gripper_slow, required_steps=10)],
       },
       logical="all",
   )

Each sequence completes independently, and ``logical="all"`` requires both to finish.
``CompletionCriteriaRunner`` remembers each sequence's completion until the episode resets.
The successful periods do not have to overlap. For example, if the object is still during steps
1–10 and the gripper is slow during steps 2–11, the criteria are complete at step 11, even if the
object is moving again then.

``logical`` combines completed sequences; it does not create separate counters inside one predicate.

Both conditions during the same steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine the instantaneous checks before wrapping them to require ten shared steps:

.. code-block:: python

   def object_and_gripper_stable(env):
       return object_still(env) & gripper_slow(env)

   criteria = CompletionCriteria(
       name="simultaneous_stability",
       predicate_sequence=[
           TrueForConsecutiveStepsCfg(
               predicate=object_and_gripper_stable,
               required_steps=10,
           ),
       ],
   )

One counter tracks the combined condition. If either check becomes false, the streak starts over.
If the object is still only during steps 1–10 and the gripper is slow only during steps 2–11,
this requirement does not complete: they overlap for only nine steps.


Subtask progress tracking in composite and sequential tasks
-----------------------------------------------------------

``CompositeTaskBase`` collects subtask criteria sets in a flat ``TaskTerminationCfg.success`` list.
It prefixes their names with ``subtask_<index>/`` and sets ``parent_subtask_idx`` to identify
which subtask each criteria set belongs to. Standalone tasks retain their original criteria names,
such as ``pick_and_place``. Nested composite or sequential tasks are not supported.

For an order-independent composite task, every subtask's criteria sets are active.
With ``CompositeTaskBase(..., subtasks_are_sequential=True)``, ``ProgressTracker``
activates each subtask only after all criteria sets of the preceding subtask complete in that
environment. The next subtask starts on the following environment step. A later subtask's predicates
cannot advance before that subtask becomes active, even if their physical conditions already happen
to be true.

``ProgressTracker`` determines task success and reports the same criteria completion history.
Completed milestones remain recorded. ``TaskTerminationCfg.desired_subtask_success_state``
preserves the composition's optional final-condition checks. Reports contain the flat criteria sets
and their weighted overall progress; subtask metrics read ``ProgressTracker.get_subtask_completion()``.
For a consecutive-step final condition, these checks continue updating its counter. If the condition
becomes false, a new streak is required, but the recorded subtask completion is kept.
There are no additional reports for parent criteria sets. See
:doc:`concept_composite_tasks_design` for composition and success semantics.

.. figure:: ../../../images/composite_vs_sequential_progress_tracking.png
   :width: 100%
   :alt: Comparison of predicate tracking activation in composite and sequential tasks
   :align: center

   Composite tasks activate tracking on all subtasks' predicates together, while sequential tasks activate
   tracking on each subtask's predicates only after the preceding subtask succeeds.


Reading subtask progress tracking at runtime
--------------------------------------------

``ProgressTrackingRecorder`` puts progress results in the environment's ``extras`` dictionary.
Read each environment's state and completed-predicate events as follows:

.. code-block:: python

   progress = env.unwrapped.extras["progress_tracking"]

   state = progress["states"][env_id]
   print(state.overall_score, state.all_complete)

   criteria = state.criteria_by_name["pick_and_place"]
   print(criteria.score, criteria.is_complete)
   print(criteria.prerequisites_met)
   print(criteria.active_predicates)

   for event in progress["events"][env_id]:
       print(event.step, event.criteria_name, event.sequence_name, event.predicate_name)

After an automatic reset, ``env.extras["progress_tracking"]`` still shows the finished episode
until the next step.

Arena's episode recorder also serializes the final progress state and predicate events into the
episode's JSONL record when an output path is configured. Tasks without completion criteria have
no success termination or progress-tracking configuration and produce no progress fields.
Each criteria set records ``prerequisites_met``. When it is false, reports show
"waiting for prerequisites" separately from pending active predicates. Waiting contributes no
score, completion events, or funnel stages. For older recordings in the same schema that omit this
field, reports assume prerequisites are met, preserving their existing display.

For example, one entry of the JSONL record may look like this (placement predicate name shortened):

.. code-block:: json

   {
     "progress": {
       "overall_score": 0.5,
       "all_complete": false,
       "criteria_by_name": {
         "pick_and_place": {
           "score": 0.5,
           "is_complete": false,
           "completed_sequences": 0,
           "total_sequences": 1,
           "prerequisites_met": true,
           "active_predicates": {
             "default_sequence": "object_on_destination"
           }
         }
       },
       "events": [
         {
           "step": 18,
           "criteria_name": "pick_and_place",
           "sequence_name": "default_sequence",
           "predicate_index": 0,
           "predicate_name": "ObjectLifted(object_name='can')",
           "score_delta": 0.5
         }
       ]
     }
   }

The object has been lifted: one of two predicates is complete, giving a score of ``0.5``.
Placement is still required. The event records when lifting completed; settling earns no progress.

The recording schema uses the same criteria and sequence names as the runtime API. Older
recordings that use ``objectives``, ``objective``, or ``group`` fields require conversion or
regeneration before they can be read by the current report tools.
