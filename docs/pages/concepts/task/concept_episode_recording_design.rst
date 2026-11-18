Recording Per-Episode Values
============================

Arena writes one JSON line for every finished episode, with its success, task progress, and the
variations drawn for it. To record a value of your own, add an **episode recorder term**. Its output
appears in each episode's line under the term's name.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Value
     - Term
   * - Known when the episode ends, such as an object's final height
     - A function ``func(env, env_id, ...)`` that returns a dictionary
   * - Builds up during the episode, such as an object's peak speed
     - A class derived from ``EpisodeRecorderTerm``

Example
-------

This term keeps the highest speed an object reached. Arena calls ``post_step()`` after every
control step, ``reset()`` when an episode starts, and the term itself when the episode ends:

.. code-block:: python

   import torch

   from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg


   class ObjectPeakSpeed(EpisodeRecorderTerm):
       """Records the highest linear speed [m/s] an object reached during the episode."""

       def __init__(self, cfg, env):
           super().__init__(cfg, env)
           self._object_name = cfg.params["object_name"]
           self._peak_speed = torch.zeros(env.num_envs, device=env.device)

       def post_step(self, env):
           velocity_w = env.arena_world.get_mean_linear_velocity_w(self._object_name)
           speed = torch.linalg.norm(velocity_w, dim=-1)
           # Update in place: Arena may step inside torch.inference_mode() and reset outside it.
           torch.maximum(self._peak_speed, speed, out=self._peak_speed)

       def reset(self, env_ids=None):
           self._peak_speed[slice(None) if env_ids is None else env_ids] = 0.0

       def __call__(self, env, env_id, object_name):
           return {"peak_speed": float(self._peak_speed[env_id])}

A task adds it under a name of its choice. An environment passes the same dictionary to
``IsaacLabArenaEnvironment(episode_recorder_terms=...)``, as the
`external environment example <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_examples/external_environments/advanced.py>`_
does.

.. code-block:: python

   class MyPickAndPlaceTask(PickAndPlaceTask):
       def get_episode_recorder_terms(self):
           return {
               "object_motion": EpisodeRecorderTermCfg(
                   func=ObjectPeakSpeed, params={"object_name": self.pick_up_object.name}
               )
           }

Each line of ``episode_results_rebuild<N>.jsonl`` in the Run's output directory then holds the
value (some fields left out):

.. code-block:: json

   {"env_id": 0, "episode_in_env": 3, "success": true, "object_motion": {"peak_speed": 0.83}}

Rates and averages are computed when the lines are read:

.. code-block:: python

   from isaaclab_arena.recording.episode_results import read_episode_records

   records = read_episode_records("episode_results_rebuild0.jsonl")
   moved_rate = sum(record["object_motion"]["peak_speed"] > 0.5 for record in records) / len(records)

Rules
-----

* A term returns a dictionary of JSON values, or ``None`` to leave itself out of the line. Convert
  tensors with ``float()`` or ``tolist()``. NaN and infinity stop the evaluation with an error that
  names the term.
* ``__call__`` and a function term declare one argument for each key in ``params``, as
  ``object_name`` above, and no ``**params``.
* Each term name is used once. The core fields (such as ``success`` and ``episode_length``) and
  ``placement``, ``variations``, and ``progress`` are reserved.
* A composite task adds ``_subtask_<i>`` to its subtasks' term names, for example
  ``object_motion_subtask_0``. A subtask's terms run for the whole episode.
