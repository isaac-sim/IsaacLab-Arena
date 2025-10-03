``isaac_arena`` Documentation
=============================

``isaac_arena`` is an extension to `Isaac Lab <https://isaac-sim.github.io/IsaacLab/main/index.html>`_
for providing an environment for robotic policy evaluation.

A key feature of ``isaac_arena`` is an easier, more composable interface for creating environments.


.. _quickstart:

Installation
============

``isaac-arena`` version ``v1.0.0`` only supports installation from source in a docker container.
See :doc:`pages/installation` for more options.

Examples
========

Below are some example environments built using ``isaac_arena``.

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
   pages/example_tabletop_manipulation
   pages/example_locomanipulation
   pages/concept_environment_design
