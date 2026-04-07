Task
====

A task defines what the robot is supposed to do.
It specifies the success criteria, episode termination conditions,
reset logic, and the metrics used to measure performance.

.. code-block:: python

   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   destination = ObjectReference("kitchen_counter", parent_asset=kitchen)

   task = PickAndPlaceTask(
       pick_up_object=pick_object,
       destination_location=destination,
       background_scene=kitchen,
   )

   environment = IsaacLabArenaEnvironment(
       name="kitchen_pick_and_place",
       embodiment=embodiment,
       scene=scene,
       task=task,
   )

Walkthrough
-----------

A task operates on the objects that are already in the scene.
In the example above, ``pick_object`` is the cracker box the robot needs to pick up,
and ``destination`` is a reference to a surface in the kitchen where it should be placed.

.. code-block:: python

   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   destination = ObjectReference("kitchen_counter", parent_asset=kitchen)

   task = PickAndPlaceTask(
       pick_up_object=pick_object,
       destination_location=destination,
       background_scene=kitchen,
   )

Under the hood, the task contributes termination conditions (e.g. object reached the goal,
or object was dropped), reset events (randomizing the object's starting position each episode),
and metrics (success rate, object moved rate) to the compiled environment.
You do not need to wire any of this up manually — it happens automatically when the environment
is compiled.

Available tasks include (but are not limited to) ``PickAndPlaceTask``, ``LiftObjectTask``, ``OpenDoorTask``, and ``CloseDoorTask``.

**RL tasks**

RL tasks extend their imitation learning counterparts with the additional components
needed for training: a command manager that samples a new goal each episode,
dense reward terms, and goal-conditioned observations.
For example, ``LiftObjectTaskRL`` extends ``LiftObjectTask`` and adds a target height
command and a reward for lifting the object toward it.

**Sequential tasks**

Tasks can be chained together using ``SequentialTaskBase``.
Each subtask must be completed in order before the next one begins.

.. code-block:: python

   pick_and_place_task = PickAndPlaceTask(pick_object, destination, kitchen)

   openable_object = OpenableObjectReference(
       "refrigerator", parent_asset=kitchen, openable_joint_name="fridge_door_joint"
   )
   close_door_task = CloseDoorTask(openable_object, closedness_threshold=0.10)

   task = SequentialTaskBase(subtasks=[pick_and_place_task, close_door_task])

More details
------------

The rest of this section covers further details of the task component.

.. toctree::
   :maxdepth: 1

   concept_sequential_tasks_design
   concept_metrics_design
