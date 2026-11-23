Environment
===========

Isaac Lab Arena aims to simplify the creation of task/environment libraries.
The key to achieving that goal is the use of *composition*.
Arena environments are composed of three independent sub-pieces:

* **Scene**: The scene is a collection of objects, backgrounds, lights, etc.
* **Embodiment**: The robot embodiment, its physical description, observations, actions, sensors etc.
* **Task**: A definition of what is to be accomplished in the environment.

.. figure:: ../../../images/isaac_lab_arena_arch_overview.png
   :width: 90%
   :alt: Isaac Lab Arena Workflow
   :align: center

   The architecture of Isaac Lab Arena. Evaluation environments are composed of
   three independent sub-pieces: Scene, Embodiment, and Task. These sub-pieces
   are passed to the Environment Compiler to produce an Isaac Lab manager-based
   environment.

In code, this looks like:

.. code-block:: python

   scene = Scene(assets=[background, pick_up_object])

   environment = IsaacLabArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,
       task=task,
       teleop_device=teleop_device,  # optional
       default_physics_backend=PhysicsBackend.PHYSX,  # default; use NEWTON when required
   )

   env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg()).make_registered()

``ArenaEnvBuilder`` compiles the scene, embodiment, and task configurations into
a single Isaac Lab ``ManagerBasedRLEnv``. The ``make_registered()`` call registers
the environment with the gym registry and returns it ready to run.

Because these pieces are independent, they can be reused and combined freely.
The same pick-and-place task works with any robot on any scene: swap the Franka
for a G1, or the kitchen for a warehouse, with no changes to the task.
This moves us from a library of monolithic environment descriptions to a library
of environment *parts*.

The pages below cover how you define an environment, how it is built, and how physics is configured:

- :doc:`environment_definition` — the two ways to define an environment, Python
  or YAML.
- :doc:`env_builder` — how ``ArenaEnvBuilder`` compiles a specification into an
  Isaac Lab ``ManagerBasedRLEnv``.
- :doc:`physics_configuration` — where physics settings belong and when they are applied.
- :doc:`physics_backend_selection` — PhysX vs Newton, ``--presets``,
  and ``default_physics_backend``.
- :doc:`env_cfg_override` — graph YAML ``env_cfg_override`` and Hydra ``_target_`` rules.

The individual components are covered in :doc:`../scene/index`,
:doc:`../embodiment/index`, and :doc:`../task/index`.

.. toctree::
   :maxdepth: 1

   environment_definition
   env_builder
   physics_configuration
   physics_backend_selection
   env_cfg_override

.. _recording-per-episode-values:

Recording your own values
-------------------------

For every finished episode, Arena writes one line to the episode results, with its success, task
progress, and the variations drawn for it. An environment adds a value of its own by passing an
**episode recorder term** to ``IsaacLabArenaEnvironment``. Each episode's line then holds the term's
output under the term's name:

.. code-block:: python

   environment = IsaacLabArenaEnvironment(
       ...,
       episode_recorder_terms={
           "object_peak_speed": EpisodeRecorderTermCfg(
               func=ObjectPeakSpeed, params={"object_name": pick_up_object.name}
           ),
       },
   )

.. code-block:: json

   {"env_id": 0, "episode_in_env": 3, "success": true, "object_peak_speed": 0.83}

A term returns a JSON value, such as a number or a dictionary, when the episode ends. Write it as:

* **a function**, for a value you can read at the end of the episode, such as an object's final
  height:

  .. code-block:: python

     def record_object_final_height(env, env_id: int, object_name: str) -> float:
         """Records the height [m] of an object when the episode ends."""
         return float(env.arena_world.get_pose_w(object_name)[env_id, 2])

* **a class derived from** ``EpisodeRecorderTerm``, for a value that builds up during the episode,
  such as an object's peak speed. Arena calls its ``post_step()`` after every step, so it can track
  the value as the episode runs.

If a value is NaN or infinite, for example because the simulation became unstable, Arena writes ``null``
and prints a warning that names the term.

:doc:`../../arena_in_your_repo/external_tasks_and_embodiments` adds ``record_object_final_height`` and
Arena's ``ObjectPeakSpeed`` to an environment.
