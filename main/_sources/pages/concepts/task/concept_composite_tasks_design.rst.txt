Composite and Sequential Tasks
==============================

An Arena task describes what the robot should do in an environment, such as opening a door or placing an object in a bin.
Arena can combine multiple tasks (subtasks) into one longer-horizon task in two ways:

* ``CompositeTaskBase`` creates an **order-independent** task. Its subtasks may succeed in any order.
* ``SequentialTaskBase`` creates an **ordered** task. It is a ``CompositeTaskBase`` subclass that
  requires subtasks to succeed in the listed order.

Both classes collect the subtasks scene configuration, reset events, failure terminations,
metrics, Mimic configuration, and progress objectives. They replace the individual success
terminations with one success condition for the complete task.

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
     - Every subtask must report success at least once. A subtask may return to an unsuccessful
       state after it has been completed.
   * - ``SequentialTaskBase``
     - List order
     - Each subtask must report success after the preceding subtask has completed. A completed
       subtask may subsequently return to an unsuccessful state.


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

By default, both composite and sequential tasks only require every subtask to have succeeded at some point. Use
``desired_subtask_success_state`` if you want the final simulator state to be a specific one:

.. code-block:: python

   task = SequentialTaskBase(
       subtasks=[pick_and_place_task, close_door_task],
       desired_subtask_success_state=[True, True],
   )

Each entry corresponds to one subtask with ordering corresponding to the order of the subtask list in the definition:

* ``True`` requires the subtask to have succeeded previously and to be succeeding now.
* ``False`` requires the subtask to have succeeded previously but to be unsuccessful now.
* ``None`` ignores that subtask when checking the final state.
