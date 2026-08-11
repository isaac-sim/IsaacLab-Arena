Pick and Place on a Kitchen Countertop
=======================================

This example uses the agentic environment-generation system to infer an environment graph spec from a prompt
for **pick and place atomic task on a kitchen countertop** in the ``lightwheel_robocasa_kitchen`` background.

Environment Description
-----------------------

.. figure:: ../../../../images/agentic_environment_generation/agentic_ui_kitchen_pnp_prompt_robot.png
   :width: 100%
   :alt: Generated kitchen pick and place task shown in the agentic environment-generation GUI.
   :align: center

   The generated kitchen scene: a DROID arm standing on the floor next to the counter, a mustard bottle and
   a bowl on the countertop. The task is to pick up the mustard bottle and place it in the bowl.

The agent picks out prims *inside* the kitchen background — a countertop surface and the floor — and emits them as
``object_references``. Arena's relation solver then places the objects on the referenced countertop and the
robot on the floor next to it.

The generated spec for this example is available at
``isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml``.

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
