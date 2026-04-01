Sequential Tasks Design
=======================

Tasks can be composed sequentially to form longer horizon, more complex tasks using the ``SequentialTaskBase`` class.
``SequentialTaskBase`` takes a list of ``TaskBase`` instances and automatically composes them into a single task.
The order of the tasks in the list determines the order in which subtasks must be completed.

**Usage Example (Pick and Place Task and Close Door Task Composition)**

.. code-block:: python

    pick_object = asset_registry.get_asset_by_name("mustard_bottle")()
    destination = ObjectReference("refrigerator_shelf", parent_asset=kitchen)
    pick_and_place_task = PickAndPlaceTask(pick_object, destination, kitchen)

    openable_object = OpenableObjectReference("refrigerator", parent_asset=kitchen, openable_joint_name="fridge_door_joint")
    close_door_task = CloseDoorTask(openable_object, closedness_threshold=0.10)

    sequential_task = SequentialTaskBase(subtasks=[pick_and_place_task, close_door_task])

**Available Examples**

- **PutAndCloseDoorTask**: Pick up and move an object to a destination location and then close a door (within **GR1PutAndCloseDoorEnvironment**).
