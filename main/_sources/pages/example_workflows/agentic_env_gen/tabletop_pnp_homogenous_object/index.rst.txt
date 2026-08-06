Pick and Place atomic task with homogeneous objects
===================================================

This example uses the agentic environment-generation system to infer an environment graph spec from a prompt
for **pick and place atomic task with homogeneous objects** on a table-top scene.

Environment Description
-----------------------

.. figure:: ../../../../images/tabletop_agentic_env_banana_bagel_plate.png
   :width: 100%
   :alt: Generated table-top pick and place task shown in the agentic environment-generation GUI.
   :align: center

   The generated table-top scene: a DROID arm, a banana, a plate, two bagels and a bowl. The task is to pick
   up the banana and place it on the plate.


The scene is *homogeneous* because each parallel environment has the same objects, embodiment, background scene, spatial relationships and task.

The generated spec for this example is available at
``isaaclab_arena_environments/maple_table_top/droid_banana_on_plate_maple_table.yaml``.

Workflow
--------

Prerequisites
^^^^^^^^^^^^^

See :ref:`agentic-env-gen-prerequisites` for the container and API key setup.

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
