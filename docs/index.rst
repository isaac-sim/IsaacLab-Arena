``Isaac Lab Arena`` Documentation
=================================

``Isaac Lab Arena`` is an extension to `Isaac Lab <https://isaac-sim.github.io/IsaacLab/main/index.html>`_
for providing an environment for robotic policy evaluation.

Isaac Lab Arena is a comprehensive robotics simulation framework that enhances NVIDIA Isaac Lab by providing a composable,
scalable system for creating diverse simulation environments and evaluating robot learning policies.
The framework enables researchers and developers to rapidly prototype and test robotic tasks with various robot embodiments,
objects, and environments.

A key feature of ``Isaac Lab Arena`` is an easier, more composable interface for creating environments.


.. _quickstart:

Installation
============

``isaaclab_arena`` version ``v0.1`` only supports installation from source in a docker container.

Examples
========

Below are some example environments built using ``isaaclab_arena``.


 .. list-table::
    :class: gallery
    :widths: auto

    * - .. image:: images/g1_galileo_arena_box_pnp_locomanip.gif
          :height: 400px
          :target: pages/workflows/gr00t_workflows.html
    * - .. image:: images/kitchen_gr1_arena.gif
          :height: 400px
          :target: pages/workflows/gr00t_workflows.html


License
-------
This code is under an `open-source license <https://github.com/isaac-sim/isaac_arena/blob/main/LICENSE.md>`_ (Apache 2.0).

.. toctree::
   :maxdepth: 1
   :caption: User's Guide

   pages/quickstart/installation

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   pages/concepts/concept_affordances_design
   pages/concepts/concept_assets_design
   pages/concepts/concept_embodiment_design
   pages/concepts/concept_environment_design
   pages/concepts/concept_environment_compilation
   pages/concepts/concept_metrics_design
   pages/concepts/concept_policy_design
   pages/concepts/concept_scene_design
   pages/concepts/concept_tasks_design
   pages/concepts/concept_teleop_devices_design

.. toctree::
   :maxdepth: 1
   :caption: Sample Tasks

   pages/sample_tasks/g1_locomanipulation_box_pick_and_place_task
   pages/sample_tasks/gr1_open_microwave_task

.. toctree::
   :maxdepth: 1
   :caption: Examples

   pages/examples/create_a_new_affordance.rst
   pages/examples/create_a_new_asset.rst
   pages/examples/create_a_new_embodiment.rst
   pages/examples/create_a_new_environment.rst
   pages/examples/create_a_new_metric.rst
   pages/examples/create_a_new_task.rst
