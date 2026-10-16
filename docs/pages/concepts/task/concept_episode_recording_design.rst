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

   class ObjectPeakSpeed(EpisodeRecorderTerm):
       """Records the highest linear speed [m/s] an object reached during the episode."""

       def __init__(self, cfg, env):
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
           return {"object": object_name, "peak_linear_speed": float(self._peak_linear_speed[env_id])}

Update the state in place, as above. Arena may step the environment inside
``torch.inference_mode()`` and reset it outside, and a tensor created inside inference mode cannot
be changed in place outside it.

Adding terms to a task or an environment
----------------------------------------

A task returns its terms from ``get_episode_recorder_terms()``. The dictionary key is the name the
output is recorded under:

.. code-block:: python

   class PickAndPlaceTask(TaskBase):
       def get_episode_recorder_terms(self):
           return {
               "object_motion": EpisodeRecorderTermCfg(
                   func=ObjectPeakSpeed, params={"object_name": self.pick_up_object.name}
               )
           }

An environment adds terms through ``IsaacLabArenaEnvironment(episode_recorder_terms=...)``, in the
same form. A term name may be used only once, and it must not be the name of a core field.

How the value appears in the episode results
--------------------------------------------

Each line of ``episode_results_*.jsonl`` then carries the term's output (some fields left out):

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
   moved_rate = sum(record["object_motion"]["peak_linear_speed"] > 0.5 for record in records) / len(records)

The removed metrics used these thresholds: an object counted as moved above 0.5 m/s, and a door
counted as moved when its openness left the reset position by more than 0.05.
