Recording Per-Episode Values
============================

Arena writes one JSON line for every finished episode. Each line holds the core fields, such as
``success`` and ``episode_length``, and the output of every episode recorder term. Tasks and
environments add their own terms to record anything else about an episode.

Each term's output is stored in the record under the term's name. Values across episodes, such
as rates and averages, are computed when the records are read.

Recording a value at the end of the episode
-------------------------------------------

A term can be a function. Arena calls it once, when the episode ends, and stores the returned
dictionary under the term's name. Return ``None`` to leave the term out of that record.

.. code-block:: python

   def record_final_height(env, env_id: int, object_name: str) -> dict:
       """Records an object's height at the end of the episode."""
       height = env.arena_world.get_pose_w(object_name)[env_id, 2]
       return {"height": float(height)}

Recording a value built up over the episode
-------------------------------------------

A term can also be a class that derives from ``EpisodeRecorderTerm``. It keeps one value per
environment and updates it after every control step in ``post_step()``. Arena calls ``reset()``
when an episode starts and calls the term when the episode ends.

The built-in ``ObjectPeakSpeed`` term records the highest speed an object reached:

.. code-block:: python

   import math
   import torch

   from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg


   class ObjectPeakSpeed(EpisodeRecorderTerm):
       """Records the highest linear speed [m/s] an object reached during the episode."""

       def __init__(self, cfg: EpisodeRecorderTermCfg, env):
           super().__init__(cfg, env)
           self._object_name = cfg.params["object_name"]
           self._peak_linear_speed = torch.zeros(env.num_envs, device=env.device)

       def post_step(self, env):
           linear_velocity_w = env.arena_world.get_mean_linear_velocity_w(self._object_name)
           linear_speed = torch.linalg.norm(linear_velocity_w, dim=-1)
           torch.maximum(self._peak_linear_speed, linear_speed, out=self._peak_linear_speed)

       def reset(self, env_ids=None):
           self._peak_linear_speed[slice(None) if env_ids is None else env_ids] = 0.0

       def __call__(self, env, env_id, object_name):
           peak_linear_speed = float(self._peak_linear_speed[env_id])
           # NaN and infinity cannot be recorded, so a value that is not finite becomes null.
           if not math.isfinite(peak_linear_speed):
               peak_linear_speed = None
           return {"object": object_name, "peak_linear_speed": peak_linear_speed}

Update the state in place, as above. Arena may step the environment inside
``torch.inference_mode()`` and reset it outside, and a tensor created inside inference mode cannot
be changed in place outside it.

A subclass declares ``__call__(self, env, env_id, ...)`` with one named parameter for each key in the
term's ``params``, as ``object_name`` above. It must not declare ``**params``.

What a term may return
----------------------

A term returns a dictionary of JSON values, or ``None`` to leave itself out of the record. The
values can be numbers, strings, booleans, ``None`` (written as ``null``), lists, and dictionaries.
Convert tensors first, with ``float()`` or ``tolist()``.

Arena does not accept NaN or infinity. A term that returns one stops the evaluation with an error
that names the term.

Adding terms to a task or an environment
----------------------------------------

A task returns its terms from ``get_episode_recorder_terms()``. The dictionary key is the name the
output is recorded under:

.. code-block:: python

   from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
   from isaaclab_arena.recording.motion_terms import ObjectPeakSpeed


   class PickAndPlaceTask(TaskBase):
       def get_episode_recorder_terms(self):
           return {
               "object_motion": EpisodeRecorderTermCfg(
                   func=ObjectPeakSpeed, params={"object_name": self.pick_up_object.name}
               )
           }

An environment adds terms through ``IsaacLabArenaEnvironment(episode_recorder_terms=...)``, in the
same form.

A term name may be used only once. These names are reserved:

* The core fields: ``job_name``, ``env_id``, ``episode_in_env``, ``seed``, ``success``,
  ``episode_length``, ``language_instruction``, ``timestamp``, and ``replay_source_record_index``.
* The terms Arena adds to every environment: ``placement``, ``variations``, and ``progress``.

Built-in motion terms
---------------------

Arena adds these terms to its tasks:

.. list-table::
   :header-rows: 1
   :widths: 18 16 26 40

   * - Term class
     - Recorded as
     - Added by
     - Fields
   * - ``ObjectPeakSpeed``
     - ``object_motion``
     - Pick and place, assembly, gear insertion, place upright, and goal pose
     - ``object``: the object's name. ``peak_linear_speed``: the highest linear speed [m/s],
       sampled once per control step.
   * - ``JointOpennessRange``
     - ``joint_motion``
     - Open door and close door
     - ``object``: the object's name. ``min_openness`` and ``max_openness``: the lowest and highest
       normalized joint position from ``Openable.get_openness``, from 0 to 1.

Composite tasks built from these tasks record the same terms, with the suffix described below.

A value that is not finite is recorded as ``null``. ``joint_motion`` is left out of the record for an
episode that ended before its first control step.

How the value appears in the episode results
--------------------------------------------

Arena writes one ``episode_results_rebuild<N>.jsonl`` file for each environment rebuild, in the
Run's output directory. The policy runner writes ``episode_results_rank<N>.jsonl`` instead. Each line
then carries the term's output (some fields left out):

.. code-block:: json

   {"env_id": 0, "episode_in_env": 3, "success": true,
    "object_motion": {"object": "cracker_box", "peak_linear_speed": 0.83}}

The value also appears in ``arena_experiment_result.json`` and as episode metadata on the Run
pages of the evaluation report.

Composite tasks
---------------

A composite task collects the terms of its subtasks and adds ``_subtask_<i>`` to their names. For
the sequential ``put_item_in_fridge_and_close_door`` task, which picks and places an object and
then closes the refrigerator door, a record contains:

.. code-block:: json

   {"success": true,
    "object_motion_subtask_0": {"object": "ranch_dressing_hope_robolab", "peak_linear_speed": 0.41},
    "joint_motion_subtask_1": {"object": "refrigerator", "min_openness": 0.02, "max_openness": 0.71}}

A subtask's terms run for the whole episode, not only while that subtask is active. When each
subtask finished is recorded in the ``progress`` field.

Replacing a removed metric
--------------------------

Before Arena removed metrics, ``ObjectMovedRateMetric`` needed three pieces: a recorder term that
saved the object's velocity for every step into an HDF5 file, a function that read the data back
after the rollout, and a ``MetricBase`` class with a ``MetricTermCfg`` that tied the two together:

.. code-block:: python

   class ObjectVelocityRecorder(RecorderTerm):
       def record_post_step(self):
           # Saved for every step of every episode, into the HDF5 file.
           return self.name, self._env.arena_world.get_mean_linear_velocity_w(self.object_name)


   def compute_object_moved_rate(recorded_metric_data, object_velocity_threshold):
       # Runs after the rollout, on the data read back from the HDF5 file.
       object_moved_per_demo = []
       for object_velocity in recorded_metric_data:
           speed = np.linalg.norm(object_velocity, axis=-1)
           object_moved_per_demo.append(np.any(speed > object_velocity_threshold))
       return np.mean(object_moved_per_demo)

``ObjectPeakSpeed`` and the calculation below do the same job without storing every step and
without an HDF5 file.

Computing a rate across episodes
--------------------------------

Read the records and compare the raw values against a threshold:

.. code-block:: python

   import json

   with open("episode_results_rebuild0.jsonl", encoding="utf-8") as f:
       records = [json.loads(line) for line in f]
   peak_speeds = [record["object_motion"]["peak_linear_speed"] for record in records]
   # A speed that was not finite is recorded as null. Leave those episodes out of the rate.
   known_peak_speeds = [speed for speed in peak_speeds if speed is not None]
   moved_rate = sum(speed > 0.5 for speed in known_peak_speeds) / len(known_peak_speeds)

The removed metrics used these thresholds: an object counted as moved above 0.5 m/s, and a door
counted as moved when its openness left the reset position by more than 0.05.

For a door, compare ``min_openness`` and ``max_openness`` with the reset openness. The record does not
carry the reset openness. It is the task's ``reset_openness``, which is 0.0 by default for
``OpenDoorTask`` and 1.0 by default for ``CloseDoorTask``. The door moved if ``max_openness`` is more
than 0.05 above it or ``min_openness`` is more than 0.05 below it.
