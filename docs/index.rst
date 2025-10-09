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

``isaaclab_arena`` version ``v1.0.0`` only supports installation from source in a docker container.
See :doc:`pages/installation` for more options.

Examples
========

Below are some example environments built using ``isaaclab_arena``.

.. TODO::

   (alexmillane, 2025-10-03): Add examples.

.. .. list-table::
..     :class: gallery
..     :widths: auto

..     * - .. image:: images/3dmatch.gif
..          :height: 200px
..          :target: pages/torch_examples_reconstruction.html
..       - .. image:: images/desk_radio_x2_600px.gif
..          :height: 200px
..          :target: pages/torch_examples_deep_features.html

License
-------
This code is under an `open-source license <https://github.com/isaac-sim/isaac_arena/blob/main/LICENSE.md>`_ (Apache 2.0).

.. TODO::
   (alexmillane, 2025-10-03): Confirm license.

.. toctree::
   :maxdepth: 1
   :caption: User's Guide

   pages/installation

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   pages/concepts/concept_affordances_design
   pages/concepts/concept_assets_design
   pages/concepts/concept_embodiment_design
   pages/concepts/concept_environment_design
   pages/concepts/concept_metrics_design
   pages/concepts/concept_policy_design
   pages/concepts/concept_scene_design
   pages/concepts/concept_tasks_design
   pages/concepts/concept_teleop_devices_design

.. toctree::
   :maxdepth: 1
   :caption: Examples

   pages/examples/example_locomanipulation
   pages/examples/example_tabletop_manipulation
