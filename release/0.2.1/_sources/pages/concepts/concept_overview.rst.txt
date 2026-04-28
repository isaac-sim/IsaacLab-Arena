Concept Overview
================

Isaac Lab Arena aims to simplify the creation of task/environment libraries.
The key to achieving that goal is the use of *composition*.
Arena environments are composed of three independent sub-pieces:

* **Scene**: The scene is a collection of objects, backgrounds, lights, etc.
* **Embodiment**: The robot embodiment, its physical description, observations, actions, sensors etc.
* **Task**: A definition of what is to be accomplished in the environment.

.. figure:: ../../images/isaac_lab_arena_arch_overview.png
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
   )

   env = ArenaEnvBuilder(environment, args_cli).make_registered()

``ArenaEnvBuilder`` compiles the scene, embodiment, and task configurations into
a single Isaac Lab ``ManagerBasedRLEnv``. The ``make_registered()`` call registers
the environment with the gym registry and returns it ready to run.

Because these pieces are independent, they can be reused and combined freely.
The same pick-and-place task works with any robot on any scene: swap the Franka
for a G1, or the kitchen for a warehouse, with no changes to the task.
This moves us from a library of monolithic environment descriptions to a library
of environment *parts*.
