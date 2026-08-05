Agentic Environment Generation and Policy Evaluation
====================================================

Agentic environment generation creates Arena environments from natural-language
prompts. It resolves the prompt into ``ArenaEnvGraphSpec`` by the agent, which specifies the scene layout, tasks, and spatial relations.
This spec is then used to compose the scene and build the environment.
The environment can be used for policy evaluation.

.. todo:: add concept overview page


In this section, we will walk through the following example environment generation workflows to explain how to use this tool for your own tasks.

- Table-top Pick and Place task

  - :doc:`tabletop_pnp_homogenous_object/index`
  - :doc:`tabletop_pnp_heterogeneous_object/index`
  - :doc:`tabletop_pnp_composite_task/index`

- Kitchen Pick and Place task

  - :doc:`kitchen_pick_and_place`

- Kitchen Open/Close Door task

  - :doc:`kitchen_open_door`

Available Generated Specs
-------------------------

The ``isaaclab_arena_environments/robolab`` subfolder contains Arena environment graph specs for
RoboLab scenes and tasks. Scene YAMLs live in ``robolab/scenes/``; task YAMLs in
``robolab/tasks/`` include their scene via a top-level ``external_yaml:`` path. See
:doc:`../robolab_task_catalog` for the list of RoboLab tasks currently supported in Arena.


Warnings
--------

.. note::
   Agentic environment generation is experimental and changing quickly. The
   current prompt formats, generated spec structure, GUI behavior, and policy
   evaluation integrations may change across releases.

   We are actively working on:

   * Support for more complex scene layouts and object placements.
   * Support for more diverse task specifications.

.. toctree::
   :maxdepth: 1
   :hidden:

   tabletop_pnp_homogenous_object/index
   tabletop_pnp_heterogeneous_object/index
   tabletop_pnp_composite_task/index
   kitchen_pick_and_place
   kitchen_open_door
   gui_runner
