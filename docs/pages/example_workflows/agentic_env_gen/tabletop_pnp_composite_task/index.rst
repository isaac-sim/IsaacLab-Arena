Pick and Place composite task with SimReady assets
==================================================

This example uses the agentic environment-generation system to resolve a prompt into an environment graph spec
for **pick and place composite task with SimReady assets** on a table-top scene.

Environment Description
-----------------------

.. figure:: ../../../../images/tabletop_agentic_env_cans_simready.png
   :width: 100%
   :alt: Agentic environment-generation GUI showing a table-top pick and place composite task with a DROID arm,
      a pepsi can, a tuna can, a mini plastic basket, a hammer and a bean can.
      The task is to pick up the pepsi can and bean can from the maple table and place them into the mini plastic basket.
   :align: center

The task is *composite*: instead of a single pick and place, the robot picks up
both the pepsi can and the bean can from the maple table and places them into
the mini plastic basket. Two atomic ``PickAndPlaceTask`` subtasks are combined
under one root task. A hammer and a tuna can are on the table as distractors.

Two of the assets (i.e. pepsi can and plastic basket) are not in the Arena asset catalog. They are resolved by
asset search through SimReady service and entered the spec as ``simready_usd_object`` entries carrying a ``usd_path``.

The resolved spec for this example is available at
``isaaclab_arena_environments/maple_table_top/simready_droid_pick_place_cans_hammer_maple_table.yaml``.

Workflow
--------

Prerequisites
^^^^^^^^^^^^^

**Docker Container**: Base (see :doc:`../../../quickstart/installation` for more details)

:docker_run_default:

The generation agent calls a remote LLM endpoint, so export your API key inside
the container before launching the runner:

.. code-block:: bash

   export NV_API_KEY=<your-api-key>

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
