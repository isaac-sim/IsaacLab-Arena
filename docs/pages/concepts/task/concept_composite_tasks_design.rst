Composite and Sequential Tasks
==============================

``CompositeTaskBase`` combines tasks, such as placing an object and closing a door:

* By default, subtasks may finish in any order.
* ``subtasks_are_sequential=True`` requires them to finish in the listed order.

The class combines the subtasks' scene settings, reset events, and metrics.
Its ``get_termination_cfg()`` returns one ``TaskTerminationCfg`` with the subtasks' success
objectives, failure conditions, and a time limit for the whole task.


By default, a finished subtask stays marked complete. For example, placing the can still counts
if it moves while the robot places the bottle. Use ``desired_subtask_success_state`` below to
require the can to be in place when the whole task finishes.


Composing tasks
---------------

Pass a flat list of ``TaskBase`` instances. Composite tasks cannot contain other composite tasks.
For example, let the robot place two objects in either order:

.. code-block:: python

   from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
   from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

   place_can = PickAndPlaceTask(can, bin_reference, table)
   place_bottle = PickAndPlaceTask(bottle, bin_reference, table)

   packing_task = CompositeTaskBase(
       subtasks=[place_can, place_bottle],
       task_description="Place the can and bottle into the bin.",
   )

Set ``subtasks_are_sequential=True`` when the order is part of the task. Here, placing the object must happen
before closing the refrigerator:

.. code-block:: python

   from isaaclab_arena.tasks.close_door_task import CloseDoorTask
   from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
   from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

   pick_and_place_task = PickAndPlaceTask(pick_object, refrigerator_shelf, kitchen)
   close_door_task = CloseDoorTask(refrigerator, closedness_threshold=0.10)

   put_away_task = CompositeTaskBase(
       subtasks=[pick_and_place_task, close_door_task],
       subtasks_are_sequential=True,
       task_description="Place the object in the refrigerator, then close the door.",
   )

See :doc:`../../example_workflows/sequential_static_manipulation/index` for a complete sequential-task
workflow that places an object in a refrigerator and then closes the door.


Specifying a final subtask state
--------------------------------

Use ``desired_subtask_success_state`` to check subtask conditions when the whole task finishes:

.. code-block:: python

   task = CompositeTaskBase(
       subtasks=[pick_and_place_task, close_door_task],
       subtasks_are_sequential=True,
       desired_subtask_success_state=[True, True],
   )

The entries follow the order of ``subtasks``:

* ``True`` requires the subtask to have completed and its final condition to hold now.
* ``False`` requires the subtask to have completed and its final condition to be false now.
* ``None`` ignores the subtask in the success check. In a sequential task, it must still finish
  before the next subtask starts.

For a task with one ordered predicate sequence, the final condition is the last predicate.
Earlier milestones stay recorded: a placed object does not need to remain above its initial lift
height, for example. Conditions that must hold together belong in the same final predicate.
For named sequences, the objective's ``ALL``, ``ANY``, or ``CHOOSE`` setting combines their final predicates.
If a subtask defines multiple objectives, all their final conditions must hold for its current result to be true.
