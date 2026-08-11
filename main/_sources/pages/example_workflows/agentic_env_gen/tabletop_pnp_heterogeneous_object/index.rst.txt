Pick and Place atomic task with heterogeneous objects
=====================================================

This example uses the agentic environment-generation system to infer an environment graph spec from a prompt
for **pick and place atomic task with heterogeneous objects** for table-top manipulation.

Environment Description
-----------------------

.. figure:: ../../../../images/agentic_environment_generation/tabletop_agentic_env_fruits.png
   :width: 100%
   :alt: Generated table-top pick and place task shown in the agentic environment-generation GUI.
   :align: center

   The generated table-top scene: a DROID arm, a bowl and a fruit sampled per environment from a set of fruit
   assets. The task is to pick up the fruit and place it into the bowl.

A DROID arm stands at a maple table, picks up a fruit and places it into a bowl
on the table.

The scene is *heterogeneous* because each parallel environment spawns a
different fruit, while the embodiment, background scene, spatial relationships
and task stay the same. The fruit is declared as an **object set**: a group of
interchangeable assets under a single id, distributed one member per
environment. Running with ``--num_envs 1`` shows a single fruit, so raise
``--num_envs`` to see the variation.

The generated spec for this example is available at
``isaaclab_arena_environments/maple_table_top/droid_pick_fruit_into_bowl_maple_table.yaml``.

Workflow
--------

Prerequisites
^^^^^^^^^^^^^

See :ref:`agentic-env-gen-prerequisites` for the container and API key setup.
The spec shown here depends on the model behind that endpoint — see
:doc:`../../../concepts/agentic_environment_generation/model_selection`.

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
