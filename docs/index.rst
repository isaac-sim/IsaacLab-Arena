Welcome to Isaac Lab Arena!
===========================

Isaac Lab Arena extends `Isaac Lab <https://isaac-sim.github.io/IsaacLab/main/index.html>`_
with composable tools for creating robotics simulation environments and running them efficiently at scale.

.. note::
   This is the development version of Isaac Lab Arena. It contains the newest features but may not be fully tested yet.
   For the tested version, please refer to the `release/0.2.1 branch <https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/index.html>`_.

| **Modular Environments**
| Compose scenes, embodiments, and tasks as reusable building blocks instead of duplicating full environment definitions.

.. figure:: images/variation_axis_web.webp
   :width: 80%
   :align: center


| **Swappable assets**
| Run-time swapping of building blocks such as target objects and backgrounds allows parallel evaluation covering many variations of one task without code changes.

.. container:: image-gallery gallery-3col

   .. video:: ./images/teaser_page/object_swapping/alphabet_soup_can_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/lemon_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/mug_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/mustard_bottle_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/orange_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/rubiks_cube_home_office_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/sugar_box_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/tomato_sauce_can_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/object_swapping/billiard_hall_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:


| **Automatic Object Placement**
| Define layouts with semantic spatial relations rather than hand-coded poses. Arena solves and validates candidate placements against object geometry, collisions, and task constraints.


.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item::
      :columns: 12 12 5 5

      .. container:: compact-code

         .. code-block:: python

            plate.add_relation(On(table))
            banana.add_relation(On(plate))
            bowl.add_relation(On(table))
            bagel_1.add_relation(On(table))
            bagel_2.add_relation(On(table))


   .. grid-item::
      :columns: 12 12 7 7

      .. video:: ./images/teaser_page/automatic_object_placement/bagels_on_plate_web.mp4
         :autoplay:
         :loop:
         :muted:
         :playsinline:
         :nocontrols:
         :width: 70%


| **Parallel Evaluation**
| Evaluate a policy across many parallel environments with aggregated metrics for high-level summary as well as per-episode recording for detailed analysis.

.. container:: image-gallery

   .. video:: ./images/teaser_page/parallel_evaluation/big_pumpkin_in_bin_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/parallel_evaluation/mouse_on_keyboard_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/parallel_evaluation/small_pumpkin_in_bin_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/parallel_evaluation/mustard_in_left_bin_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:


| **Agentic environment creation**
| Turn a natural-language task request into a runnable Arena environment. Edit the environment specification and review the randomized layouts in a GUI interactively.

.. container:: image-gallery

   .. video:: ./images/teaser_page/agentic_environment_creation/penisula_mustard_mesh_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. image:: ./images/teaser_page/agentic_environment_creation/object-ref-replicator-counter_web.webp


| **Built-in Evaluation Environments**
| Run policies against built-in registered environments.

.. container:: image-gallery gallery-3col

   .. image:: ./images/teaser_page/built_in_evaluation_environments/big_pumpkin_in_bin_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/bagels_on_plate_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/canned_food_in_bin_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/mouse_on_keyboard_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/rubiks_cube_and_banana_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/bbq_sauce_in_bin_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/small_pumpkin_in_bin_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/mustard_in_left_bin_web.webp

   .. image:: ./images/teaser_page/built_in_evaluation_environments/clutter_pumpkin_web.webp


| **Environmental Variations**
| Enable build-time and run-time variations to sample controlled changes such as but not limited to HDR backgrounds, light properties, camera intrinsic and extrinsic parameters, and object mass.

.. container:: image-gallery

   .. video:: ./images/teaser_page/variations/color_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/variations/hdr_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/variations/shadows_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/teaser_page/variations/temperature_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:


| **Multi-node Evaluation**
| Scale-out evaluation across multiple compute nodes.

.. figure:: images/teaser_page/multinode_evaluation/multinode_evaluation.png
   :width: 80%
   :align: center


| **Sensitivity Analysis**
| Analyze which environment factors are associated with policy success or failure.

.. figure:: images/sensitivity_report_200_trails.png
   :width: 100%
   :align: center


| **Teleoperation**
| Teleoperate robots using IsaacTeleop in Arena-defined environments.

.. container:: image-gallery

   .. video:: ./images/g1_galileo_arena_box_pnp_locomanip_trimmed_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:

   .. video:: ./images/gr1_sequential_static_manipulation_env_web.mp4
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:


License
========
This code is under an `open-source license <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/LICENSE.md>`_ (Apache 2.0).

Acknowledgments
===============
NVIDIA Isaac Lab-Arena is an open-source framework, available on GitHub, that provides a collaborative system for
large-scale robot policy evaluation and benchmarking in simulation, with the evaluation and task layers designed
in close collaboration with `Lightwheel <https://lightwheel.ai/>`_.

Isaac Lab-Arena was built in collaboration with the authors of Robolab
(`website <https://research.nvidia.com/labs/srl/projects/robolab/>`_, `paper <https://arxiv.org/abs/2604.09860>`_).

Contributing
============

For more details on how to contribute to Isaac Lab Arena, please refer to the
`Contributing Guidelines <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTING.md>`_.

TABLE OF CONTENTS
=================

.. toctree::
   :maxdepth: 1

   Home <self>

.. toctree::
   :maxdepth: 1
   :caption: Isaac Lab Arena Overview

   pages/motivation/motivation

.. toctree::
   :maxdepth: 1
   :caption: Set Up

   pages/quickstart/installation

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   pages/quickstart/arena_env
   pages/quickstart/arena_experiment
   pages/quickstart/environment_variations
   pages/quickstart/running_a_real_policy/index

.. toctree::
   :maxdepth: 2
   :caption: Arena in Your Repo

   pages/arena_in_your_repo/index

.. toctree::
   :maxdepth: 1
   :caption: Example Workflows

   pages/example_workflows/example_environments
   pages/example_workflows/analysis/index
   pages/example_workflows/imitation_learning/index
   pages/example_workflows/reinforcement_learning_workflows/index
   pages/example_workflows/agentic_env_gen/index

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   pages/concepts/environment/index
   pages/concepts/agentic_environment_generation/index
   pages/concepts/scene/index
   pages/concepts/task/index
   pages/concepts/embodiment/index
   pages/concepts/concept_object_and_robot_placement
   pages/concepts/policy/index
   pages/concepts/concept_arena_experiments
   pages/concepts/variations/index
   pages/concepts/concept_sensitivity_analysis

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   pages/advanced/private_omniverse
   pages/advanced/assets_management
   pages/quickstart/jupyter_notebooks

.. toctree::
   :maxdepth: 1
   :caption: References

   pages/references/release_notes
   pages/references/citing_us
