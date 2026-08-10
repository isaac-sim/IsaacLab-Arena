Pick and Place composite task with SimReady assets
==================================================

This example uses the agentic environment-generation system to infer an environment graph spec from a prompt
for **pick and place composite task with SimReady assets** on a table-top scene.

Environment Description
-----------------------

.. figure:: ../../../../images/agentic_environment_generation/tabletop_agentic_env_cans_simready.png
   :width: 100%
   :alt: Generated table-top pick and place composite task shown in the agentic environment-generation GUI.
   :align: center

   The generated table-top scene: a DROID arm, a pepsi can, a tuna can, a mini plastic basket, a hammer and a
   bean can. The task is to pick up the pepsi can and bean can from the maple table and place them into the
   mini plastic basket.

The task is *composite*: instead of a single pick and place, the robot picks up
both the pepsi can and the bean can from the maple table and places them into
the mini plastic basket. Two atomic ``PickAndPlaceTask`` subtasks are combined
under one root task. A hammer and a tuna can are on the table as distractors.

Two of the assets (i.e. pepsi can and plastic basket) are not in the Arena asset catalog. They are found by
asset search through the SimReady service and enter the spec as ``simready_usd_object`` entries carrying a ``usd_path``.

The generated spec for this example is available at
``isaaclab_arena_environments/maple_table_top/simready_droid_pick_place_cans_hammer_maple_table.yaml``.

Workflow
--------

Prerequisites
^^^^^^^^^^^^^

See :ref:`agentic-env-gen-prerequisites` for the container and API key setup.
The spec shown here depends on the model behind that endpoint — see :doc:`../model_selection`.

Workflow Steps
^^^^^^^^^^^^^^

Follow the following steps to complete the workflow:

- :doc:`step_1_launch_runner`
- :doc:`step_2_edit_environment_graph_spec`
- :doc:`step_3_use_environment`


.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_launch_runner
   step_2_edit_environment_graph_spec
   step_3_use_environment
