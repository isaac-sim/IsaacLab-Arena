Sequential Tasks
================

Tasks can be composed sequentially to form longer horizon, more complex tasks using the ``SequentialTaskBase`` class.
``SequentialTaskBase`` takes a list of ``TaskBase`` instances and automatically composes them into a single task.
The order of the tasks in the list determines the order in which subtasks must be completed.
Internally, the Arena Environment Builder will automatically track that each sub-task is
Internally, the Arena Environment Builder will automatically track that use sub-task is
completed successfully in turn.
It will also automatically compose the sub-task metrics, resets etc.

**Usage Example (Pick and Place Task and Close Door Task Composition)**

.. code-block:: python

    pick_object = asset_registry.get_asset_by_name("mustard_bottle")()
    destination = ObjectReference("refrigerator_shelf", parent_asset=kitchen)
    pick_and_place_task = PickAndPlaceTask(pick_object, destination, kitchen)

    openable_object = OpenableObjectReference("refrigerator", parent_asset=kitchen, openable_joint_name="fridge_door_joint")
    close_door_task = CloseDoorTask(openable_object, closedness_threshold=0.10)

    sequential_task = SequentialTaskBase(subtasks=[pick_and_place_task, close_door_task])

We demonstrate the use of a sequential task in one of our workflows.
See :doc:`../../example_workflows/sequential_static_manipulation/index` to see a
sequential task is use - picking up an object and placing it in a refrigerator and then
closing the door.
