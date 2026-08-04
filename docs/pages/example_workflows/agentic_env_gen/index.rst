Agentic Environment Generation and Policy Evaluation
====================================================

Agentic environment generation creates Arena environments from natural-language
prompts. It resolves the prompt into ``ArenaEnvGraphSpec`` by the agent, which specifies the scene layout, tasks, and spatial relations.
This spec is then used to compose the scene and build the environment.
The environment can be used for policy evaluation.

.. todo:: add concept overview page


In this section, we will walk through the following example environment generation workflows to explain how to use this tool for your own tasks.

- Table-top Pick and Place task
  - :doc:`tabletop_pnp_homogenous_object`
  - :doc:`tabletop_pnp_heterogeneous_object`
  - :doc:`tabletop_pnp_composite_task`
  - :doc:`tabletop_pnp_reachability_constraints`
- Kitchen Pick and Place task
- Kitchen Open/Close Door task

Available Generated Specs
-------------------------

The ``isaaclab_arena_environments/robolab`` subfolder contains Arena environment graph specs for
RoboLab scenes and tasks. Scene YAMLs live in ``robolab/scenes/``; task YAMLs in
``robolab/tasks/`` include their scene via a top-level ``external_yaml:`` path. See
:doc:`../robolab_task_catalog` for the list of RoboLab tasks currently supported in Arena.
