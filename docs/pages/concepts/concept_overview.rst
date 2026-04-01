Concept Overview
================

Isaac Lab Arena aims to simplify the creation of task/environment libraries.
The key to achieving that goal is the use of *composition*.
Arena environments are composed of three independent sub-pieces:

* **Scene**: The scene is a collection of objects, backgrounds, lights, etc.
* **Embodiment**: The robot embodiment, its physical description, observations, actions, sensors etc.
* **Task**: A definition of what is to be accomplished in the environment.

Because these pieces are independent, the can be composed in unique combinations
to create a new environments.
This is shown in the figure below.


.. figure:: ../../images/isaac_lab_arena_arch_overview.png
   :width: 90%
   :alt: Isaac Lab Arena Workflow
   :align: center

   The architecture of Isaac Lab Arena. Evaluation environment are composed of
   three independent sub-pieces: Scene, Embodiment, and Task. These sub-pieces
   are passed to the Environment Compiler to produce an Isaac Lab manager-based
   environment.

In code, this looks like:

.. code-block:: python

   # Create the scene
   scene = Scene(assets=[background, pick_up_object])

   # Arena environment
   environment = IsaacLabArenaEnvironment(
      name="manipulation_task",
      embodiment=embodiment,
      scene=scene,
      task=task,
   )

Using composition to build the environments has the advantage
that we can reuse the scenes, embodiments, and tasks in our library in many different environments.
This moves us from building a library of monolithic environment descriptions
to building a library of environment *parts*.

In this section of the documentation, we will describe the design of each of these sub-pieces:
Scene, Embodiment, and Task.

* :doc:`scene/index`

.. In the core of the workflow, we have three main components, **Scene**, **Embodiment**, and **Task**. We compile most of the managers of the
.. manager-based RL environment from these three components. We strongly incline towards keeping these components as independent as possible. This
.. allows us to reuse the components in different environments and tasks, thus making the framework more modular and easier to extend.

.. **Embodiment**

.. Embodiments define robot-specific configurations and behaviors. They provide a modular way to integrate different robots into environments,
.. encapsulating kinematics, control actions, observations, terminations and camera systems. See :doc:`./concept_embodiment_design` for more details.

.. **Task**

.. Tasks define objectives, success criteria, and behavior logic for environments. They provide configurations for termination conditions, event handling,
.. metrics collection, and mimic components. See :doc:`./concept_tasks_design` for more details.

.. **Scene**

.. Scenes manage collections of assets that define the physical environment for simulation. They provide a unified interface for composing backgrounds,
.. objects, and interactive elements. See :doc:`./concept_scene_design` for more details.

.. When combining these three components we create the observation, action, event, termination, metrics, mimic components of the manager-based RL environment.
.. For more details on how to combine these components, see :doc:`./concept_environment_compilation`.

.. Other components of interest are the **Affordances** and the **Metrics**.

.. **Affordances**

.. Affordances define what interactions objects can perform - opening doors, pressing buttons, manipulating objects.
.. They provide standardized interfaces that integrate with tasks and embodiments. See :doc:`./concept_affordances_design` for more details.

.. **Metrics**

.. Metrics define the performance evaluation metrics for the environment.
.. Some metrics are independent of the task and the embodiment, such as the success rate metric,
.. while others are task-specific, such as open door rate metric. See :doc:`./concept_metrics_design` for more details.

.. These components together with teleoperation devices form the manager-based RL environment.
.. See :doc:`./concept_environment_design` for more details on how these components are easily combined to create our environments.
