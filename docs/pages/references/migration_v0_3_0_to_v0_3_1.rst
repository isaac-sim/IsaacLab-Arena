Migrate from v0.3.0 to v0.3.1
=============================

Arena v0.3.1 consolidates task success, progress tracking, and termination configuration. It also
removes several compatibility imports. Update custom tasks and downstream Arena integrations as
described below.

Task completion and termination
-------------------------------

Replace ``ProgressObjective`` with ``CompletionCriteria``. Move the criteria returned by
``get_progress_objectives()`` into the ``success`` field of the ``TaskTerminationCfg`` returned by
``get_termination_cfg()``:

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
Predicate groups are no longer part of the public configuration.

Composite tasks
---------------

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

Consecutive-step requirements
-----------------------------

Wrap an instantaneous predicate in ``TrueForConsecutiveStepsCfg`` when success requires it to
remain true for multiple control steps:

.. code-block:: python

   CompletionCriteria(
       name="settled",
       predicate_sequence=[
           TrueForConsecutiveStepsCfg(
               predicate=objects_settled,
               required_steps=10,
           )
       ],
   )

The completion runner owns and resets the streak counter. The wrapped predicate should only report
the current step's Boolean result.

USD helper imports
------------------

Import USD helpers from their concrete modules instead of the
``isaaclab_arena.utils.usd`` package:

.. code-block:: python

   from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd
   from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths

Import ``ObjectType`` from its asset module:

.. code-block:: python

   from isaaclab_arena.assets.object_type import ObjectType
