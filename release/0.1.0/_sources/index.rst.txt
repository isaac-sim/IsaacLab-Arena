``Isaac Lab Arena`` Documentation
=================================

``Isaac Lab Arena`` is an extends `Isaac Lab <https://isaac-sim.github.io/IsaacLab/main/index.html>`_
to simplify the creation of task/environment libraries.


.. figure:: images/g1_galileo_arena_box_pnp_locomanip.gif
   :width: 100%
   :alt: G1 Locomanipulation Box Pick and Place Task
   :align: center

   A G1 humanoid robot performing a locomanipulation task of transporting a box to a tray.
   This environment was designed using Isaac Lab Arena.


The Problem
===========

With the advent of *generalist* robot policies, such as `GR00T <https://github.com/NVIDIA/Isaac-GR00T>`_
and `Pi_0 <https://github.com/Physical-Intelligence/openpi>`_,
there is a growing need to evaluate these policies in a variety of tasks/environments.

Traditional approaches to building task libraries suffer from significant limitations.
Each new environment variation—whether testing a different robot embodiment or swapping objects—requires tedious manual creation of a new task configuration.
This leads to redundant, unscalable tasks where most of the environment setup, scene configuration, and task logic is duplicated across variations.
As the number of robot types and objects grows, maintaining and extending such task libraries becomes increasingly impractical.


.. figure:: images/task_duplications.png
   :width: 100%
   :alt: Task duplications in a task library
   :align: center

   Task duplications in a task library.
   When evaluating policies across different robot embodiments and objects, most of the environment setup and task logic remains the same, leading to significant code duplication.



Can we simplify environment creation?
=====================================

.. figure:: images/variation_axis.png
   :width: 100%
   :alt: Axis of variation for the pick and place task
   :align: center

   Axis of variation of a pick and place task.
   Each environment differs along two axis, the robot embodiment and the object to be manipulated.
   All other aspects of the environment and the task remain the same.


Tasks in a task library are typically highly redundant.
For example, you may want to test how well a policy performs on a pick and place task,
on many different objects.
In this example, each environment differs in the object-to-be-manipulated,
but all other aspects remain the same.
For example, the scene layout, the robot, the observations, actions, rewards, etc are all
conserved across the environments.
Isaac Lab's manager-based environment API is convenient for expressing one such task,
but does not naturally support expressing this type of variation.

Isaac Lab Arena extends the manager-based interface to provide
a convenient way of expressing task variation, while benefiting from
the modularity, performance, and accuracy of Isaac Lab.


Isaac Lab Arena
===============

Isaac Lab Arena is a framework that simplifies the creation and maintenance of such task/environment libraries.
To simplify the expression of task/environment variation in Isaac Lab Arena,
we *compose* the environment on-the-fly from independent sub-pieces.
Because the sub-pieces are independent, they can be reused and independently varied.
Furthermore, because the environment is built on the fly, we never need to write and maintain
duplicate code.

.. figure:: images/isaac_lab_arena_arch_overview.png
   :width: 100%
   :alt: Isaac Lab Arena Architecture Overview
   :align: center

Isaac Lab Arena decomposes the environment into three independent sub-pieces:

* **Scene**: The physical environment layout. The scene is a collection of objects.
* **Embodiment**: The robot embodiment, its observations, actions, sensors etc.
* **Task**: A definition of what is to be accomplished in the environment.

The ``ArenaEnvBuilder`` composes the environment from these sub-pieces,
into a ``ManagerBasedRLEnvCfg`` which can be run in Isaac Lab.

Usage Example
=============

The following code snippet shows a simple example(pick up a tomato soup can and place it in the destination location) of how to set up a manager-based RL environment using ``isaaclab_arena``.

.. code-block:: python

   embodiment = asset_registry.get_asset_by_name("franka")(enable_cameras=True)
   background = asset_registry.get_asset_by_name("kitchen")()
   tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
   destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
   teleop_device = device_registry.get_device_by_name("keyboard")()

   # Compose the scene
   scene = Scene([background, tomato_soup_can])

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
      name="franka_kitchen_pickup",
      embodiment=embodiment,
      scene=scene,
      task=PickAndPlaceTask(tomato_soup_can, destination, background),
      teleop_device=teleop_device,
   )

   env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
   env = env_builder.make_registered() # This will register the environment with the gym registry.

.. figure:: images/franka_kitchen_pickup.gif
   :width: 100%
   :alt: Franka Kitchen Pickup Task
   :align: center

   Franka — Kitchen Pickup Task

To get started with ``isaaclab_arena``, please finish the installation process by following the instructions in :doc:`pages/quickstart/installation` and refer to the :doc:`pages/quickstart/first_arena_env` example.

Installation
============

See our :doc:`pages/quickstart/installation` page for instructions.
Note that ``isaaclab_arena`` version ``v0.1.0`` only supports installation from source in a docker container.

Examples
========

Below are some example environments built using ``isaaclab_arena``.

.. list-table::
   :class: gallery
   :widths: auto

   * - .. figure:: images/g1_galileo_arena_box_pnp_locomanip.gif
        :height: 400px
        :target: pages/sample_tasks/g1_locomanipulation_box_pick_and_place_task.html
        :alt: G1 Locomanipulation Box Pick and Place Task
        :align: center
        :figclass: gallery-fig

        G1 — Locomanipulation: Box Pick & Place

   * - .. figure:: images/kitchen_gr1_arena.gif
        :height: 400px
        :target: pages/sample_tasks/gr1_open_microwave_task.html
        :alt: GR1 Open Microwave Task
        :align: center
        :figclass: gallery-fig

        GR1 — Open Microwave Task

Check out more of our examples environments here: `IsaacLab Arena Examples <https://github.com/isaac-sim/IsaacLab-Arena/tree/main/isaaclab_arena/examples/example_environments>`_.

License
========
This code is under an `open-source license <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/LICENSE.md>`_ (Apache 2.0).

Contributing
============
For more details, please refer to the `Contributing Guidelines <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTING.md>`_.

TABLE OF CONTENTS
=================

.. toctree::
   :maxdepth: 1
   :caption: User's Guide

   pages/quickstart/installation
   pages/quickstart/first_arena_env
   pages/quickstart/docker_containers

.. toctree::
   :maxdepth: 1
   :caption: Example Workflows

   pages/example_workflows/locomanipulation/index
   pages/example_workflows/static_manipulation/index

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   pages/concepts/concept_overview
   pages/concepts/concept_environment_design
   pages/concepts/concept_embodiment_design
   pages/concepts/concept_tasks_design
   pages/concepts/concept_scene_design
   pages/concepts/concept_metrics_design
   pages/concepts/concept_teleop_devices_design
   pages/concepts/concept_environment_compilation
   pages/concepts/concept_assets_design
   pages/concepts/concept_affordances_design
   pages/concepts/concept_policy_design

.. toctree::
   :maxdepth: 1
   :caption: References

   pages/references/release_notes
