Task
====

A task defines what the robot is supposed to do in the environment.
Concretely, a task specifies three things:

- **Success condition** — when has the robot completed the task? (e.g. is the door open?)
- **Failure condition** — when should the episode end early? (e.g. has the object fallen off the table?)
- **Reset action** — how should the scene be restored at the start of each episode? (e.g. close the door)

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
Its success condition calls ``openable.is_open()`` and its reset calls ``openable.close()``.

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

Individual tasks can be combined into order-independent or sequential composite tasks. Each task
returns a ``TaskTerminationCfg`` from ``get_termination_cfg()``, declaring its ``success`` criteria,
named ``failures``, and ``timeout_s`` time limit.

.. _recording-per-episode-values:

Recording your own values
-------------------------

For every finished episode, Arena writes one line to the episode results, with its success, task
progress, and the variations drawn for it. A task adds a value of its own by returning an
**episode recorder term** from ``get_episode_recorder_terms()``. Each episode's line then holds the
term's output under the term's name:

.. code-block:: python

   class MyPickAndPlaceTask(PickAndPlaceTask):
       def get_episode_recorder_terms(self):
           return {
               "object_peak_speed": EpisodeRecorderTermCfg(
                   func=ObjectPeakSpeed, params={"object_name": self.pick_up_object.name}
               )
           }

.. code-block:: json

   {"env_id": 0, "episode_in_env": 3, "success": true, "object_peak_speed": {"peak_speed": 0.83}}

A term returns a dictionary of JSON values when the episode ends. Write it as:

* **a function**, for a value you can read at the end of the episode, such as an object's final
  height.
* **a class derived from** ``EpisodeRecorderTerm``, for a value that builds up during the episode,
  such as an object's peak speed. Arena calls its ``post_step()`` after every step, so it can track
  the value as the episode runs. ``ObjectPeakSpeed`` in
  :doc:`../../arena_in_your_repo/external_tasks_and_embodiments` is an example.

An environment adds terms the same way through ``IsaacLabArenaEnvironment(episode_recorder_terms=...)``,
and a composite task adds ``_subtask_<i>`` to its subtasks' term names. Rates and averages across
episodes are computed when the lines are read.

More details
------------

.. toctree::
   :maxdepth: 1

   concept_composite_tasks_design
   concept_progress_tracking_design
   concept_rl_tasks_design
