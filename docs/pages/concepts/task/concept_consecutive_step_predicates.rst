Consecutive-step predicates
===========================

Use ``TrueForConsecutiveStepsCfg`` in a task's ``CompletionCriteria`` to require a condition
for a number of consecutive simulation control steps. The wrapped predicate returns one
Boolean per environment. A true result increments its counter; a false result resets it.

.. code-block:: python

   from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
   from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

The examples below define three different task requirements. ``object_still(env)`` and ``gripper_slow(env)``
are configured instantaneous checks that each return one Boolean per environment.
Step numbers start when the criteria become active.

One condition after another
---------------------------

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
--------------------------------------

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
-------------------------------------

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

Episode results
---------------

The episode report shows each requirement's recorded count, target, first satisfied step,
and sequence status. See :ref:`consecutive-step-episode-summary` for the JSON fields,
HTML table, and example results.
