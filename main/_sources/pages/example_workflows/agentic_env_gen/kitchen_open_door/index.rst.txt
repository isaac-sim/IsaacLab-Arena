Open a Kitchen Fridge Door
==========================

This example uses the agentic environment-generation system to infer an environment graph spec from a prompt
for **an articulated open-door atomic task** in the ``lightwheel_robocasa_kitchen`` background.

Environment Description
-----------------------

.. figure:: ../../../../images/agentic_environment_generation/agentic_ui_kitchen_open_door.png
   :width: 100%
   :alt: Generated kitchen fridge-opening task shown in the agentic environment-generation GUI.
   :align: center

   The generated kitchen scene: a DROID arm standing on the floor next to the fridge and facing it. The task
   is to open the fridge door past an openness threshold.

The agent picks out prims *inside* the kitchen background — the floor and the fridge — and emits them as ``object_references``.
The fridge is referenced as an articulation with an openable joint, expected by the ``OpenDoorTask`` task.
Arena's relation solver places the robot on the floor next to and facing it.

The generated spec for this example is available at
``isaaclab_arena_environments/kitchen_bench/droid_open_fridge_lightwheel_kitchen.yaml``.

Workflow
--------

Prerequisites
^^^^^^^^^^^^^

See :ref:`agentic-env-gen-prerequisites` for the environment and API key setup.
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
