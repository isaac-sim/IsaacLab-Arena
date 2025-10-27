``Isaac Lab Arena`` Documentation
=================================

``Isaac Lab Arena`` is an extension to `Isaac Lab <https://isaac-sim.github.io/IsaacLab/main/index.html>`_.

Isaac Lab Arena is a comprehensive robotics simulation framework that enhances NVIDIA Isaac Lab by providing a composable,
scalable system for creating diverse simulation environments and evaluating robot learning policies.
The framework enables researchers and developers to rapidly prototype and test robotic tasks with various robot embodiments,
objects, and environments.

.. figure:: images/isaaclab_arena_core_framework.png
   :width: 100%
   :alt: Isaac Lab Arena Workflow
   :align: center

   IsaacLab Arena Workflow Overview

For a detailed overview of the workflow, please refer to the :doc:`pages/concepts/concept_overview` page.
The key components of the workflow are:

- **Scene Setup**: Define the scene layout, add asset configurations to interact with objects in interest in the scene. See :doc:`pages/concepts/concept_scene_design` for more details.
- **Embodiment Setup**: Define the robot embodiment, its observations, actions, sensors etc. See :doc:`pages/concepts/concept_embodiment_design` for more details.
- **Task Setup**: Define the task and its metrics. See :doc:`pages/concepts/concept_tasks_design` for more details.
- **Affordance Setup**: Define the affordances(interactable objects) and their interactions. See :doc:`pages/concepts/concept_affordances_design` for more details.
- **Evaluation**: Evaluate the policy and its performance in a simple and straightforward manner. See :doc:`pages/concepts/concept_metrics_design` for more details.

A key feature of ``Isaac Lab Arena`` is an easier, more composable interface for creating environments.

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

``isaaclab_arena`` version ``v0.1`` only supports installation from source in a docker container.

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

TABLE OF CONTENTS
=================

.. toctree::
   :maxdepth: 1
   :caption: User's Guide

   pages/quickstart/installation
   pages/quickstart/first_arena_env

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
