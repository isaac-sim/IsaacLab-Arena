Pick and Place atomic task with homogeneous objects
===================================================

This example uses the agentic environment-generation system to resolve a prompt into an environment graph spec
for **pick and place atomic task with homogeneous objects** on a table-top scene.

Environment Description
-----------------------

.. figure:: ../../../../images/tabletop_agentic_env_banana_bagel_plate.png
   :width: 100%
   :alt: Agentic environment-generation GUI showing a table-top pick and place task with a DROID arm, a banana, a plate, two bagels and a bowl. The task is to pick up the banana and place it on the plate.
   :align: center


The scene is *homogeneous* because each parallel environment has the same object, embodiment, background scene, spatial relationships and task.

The resolved spec for this example is available at
``isaaclab_arena_environments/maple_table_top/droid_banana_on_plate_maple_table.yaml``.

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
