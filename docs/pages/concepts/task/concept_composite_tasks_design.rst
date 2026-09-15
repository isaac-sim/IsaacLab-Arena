Composite and Sequential Tasks
==============================

An Arena task describes what the robot should do in an environment, such as opening a door or placing an object in a bin.
Arena can combine multiple tasks (subtasks) into one longer-horizon task in two ways:

* ``CompositeTaskBase`` creates an **order-independent** task. Its subtasks may succeed in any order.
* ``SequentialTaskBase`` creates an **ordered** task. It is a ``CompositeTaskBase`` subclass that
  requires subtasks to succeed in the listed order.

Both classes collect their subtasks' scene configuration, reset events, failure terminations,
metrics, and Mimic configuration. Their ``get_termination_cfg()`` combines each child's
``TaskTerminationCfg.success`` objectives under one root objective, namespaces the child failures,
and supplies the overall ``timeout_s`` budget. The environment builder uses this definition to
create the complete task's termination terms; individual subtasks do not register separate success terms.

.. note::

    "Order-independent" does not mean every subtask must be satisfied simultaneously. The environment tracks and remembers
    which subtasks have succeeded during the episode. This makes ``CompositeTaskBase`` appropriate for
    goals such as placing several objects into a bin, where the policy may choose the object order.


Choosing the composition type
-----------------------------

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Class
     - Ordering
     - Completion Criteria
   * - ``CompositeTaskBase``
     - No required order
     - Every subtask must complete its required progress. Its final physical condition may
       subsequently become false.
   * - ``SequentialTaskBase``
     - List order
     - Each subtask becomes active after the preceding subtask completes. A completed
       subtask's final physical condition may subsequently become false.


Composing tasks
---------------

Pass ordinary ``TaskBase`` instances to the composition class. For example, use ``CompositeTaskBase`` to
create an order-independent packing task for two objects:

.. code-block:: python

   from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
   from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

   place_can = PickAndPlaceTask(can, bin_reference, table)
   place_bottle = PickAndPlaceTask(bottle, bin_reference, table)

   packing_task = CompositeTaskBase(
       subtasks=[place_can, place_bottle],
       task_description="Place the can and bottle into the bin.",
   )

Use ``SequentialTaskBase`` when the order is part of the task. Here, placing the object must happen
before closing the refrigerator:

.. code-block:: python

   from isaaclab_arena.tasks.close_door_task import CloseDoorTask
   from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
   from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

   pick_and_place_task = PickAndPlaceTask(pick_object, refrigerator_shelf, kitchen)
   close_door_task = CloseDoorTask(refrigerator, closedness_threshold=0.10)

   put_away_task = SequentialTaskBase(
       subtasks=[pick_and_place_task, close_door_task],
       task_description="Place the object in the refrigerator, then close the door.",
   )

See :doc:`../../example_workflows/sequential_static_manipulation/index` for a complete sequential-task
workflow that places an object in a refrigerator and then closes the door.


Specifying a final subtask state
--------------------------------

By default, both composite and sequential tasks require every subtask to complete its progress.
Use ``desired_subtask_success_state`` to add requirements on the final simulator state:

.. code-block:: python

   task = SequentialTaskBase(
       subtasks=[pick_and_place_task, close_door_task],
       desired_subtask_success_state=[True, True],
   )

Each entry corresponds to one subtask with ordering corresponding to the order of the subtask list in the definition:

* ``True`` requires the subtask to have completed and its final condition to hold now.
* ``False`` requires the subtask to have completed and its final condition to be false now.
* ``None`` adds no final-state requirement. The subtask must still complete its progress.

For an atomic task with an ordered predicate chain, the final condition is the last predicate.
Earlier milestones stay recorded: a placed object does not need to remain above its initial lift
height, for example. Conditions that must hold together belong in the same final predicate.
