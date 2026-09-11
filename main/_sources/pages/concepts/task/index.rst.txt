Task
====

A task defines what the robot is supposed to do in the environment.
Concretely, a task specifies four things:

- **Success condition** — when has the robot completed the task? (e.g. is the door open?)
- **Failure condition** — when should the episode end early? (e.g. has the object fallen off the table?)
- **Reset action** — how should the scene be restored at the start of each episode? (e.g. close the door)
- **Metrics** — what should be measured? (e.g. how far open did the microwave get?)

.. code-block:: python

   microwave = asset_registry.get_asset_by_name("microwave")()

   task = OpenDoorTask(openable_object=microwave)

   environment = IsaacLabArenaEnvironment(
       name="open_microwave",
       embodiment=embodiment,
       scene=scene,
       task=task,
   )

Tasks and affordances
---------------------

Tasks are defined in terms of object **affordances**, not specific objects.
``OpenDoorTask`` takes any ``Openable`` — a microwave, a fridge, a cabinet.
Its success condition calls ``openable.is_open()``, its reset calls ``openable.close()``,
and its metric tracks ``openable.get_openness()``.

.. figure:: ../../../images/open_door_task.png
   :width: 100%
   :alt: OpenDoorTask defined in terms of the Openable affordance
   :align: center

   ``OpenDoorTask`` is defined entirely in terms of the ``Openable`` affordance interface,
   making it reusable with any openable object.

This is what makes tasks modular — the same task works across different objects
and scenes without any changes.

Available tasks include (but are not limited to) ``PickAndPlaceTask``, ``LiftObjectTask``,
``OpenDoorTask``, ``CloseDoorTask``, and ``PressButtonTask``.

Individual tasks can be combined into order-independent or sequential composite tasks. Tasks can also
define predicate-based progress objectives to report partial completion without changing their
success termination.

More details
------------

.. toctree::
   :maxdepth: 1

   concept_composite_tasks_design
   concept_progress_tracking_design
   concept_rl_tasks_design
   concept_metrics_design
