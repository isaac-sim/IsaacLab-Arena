Open a Kitchen Fridge Door
==========================

This example uses the agentic environment-generation GUI to create a DROID
fridge-opening task in the ``lightwheel_robocasa_kitchen`` background. The
environment-generation agent identifies the kitchen floor and fridge
articulation, while Arena's relation solver places the robot on the floor next
to and facing the fridge.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Generate the Environment
------------------------

Start the agentic environment-generation GUI:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

Enter the prompt:

.. code-block:: text

   There is a floor and a fridge in the lightwheel_robocasa_kitchen kitchen.
   DROID is on the floor, next to the fridge with 0.1 meter distance and facing
   it. DROID opens the fridge door to the 0.2 openness threshold.

.. figure:: ../../../images/agentic_ui_kitchen_open_door.png
   :width: 100%
   :alt: Agentic environment-generation GUI showing the DROID kitchen fridge-opening task
   :align: center

   The prompt generates the floor and fridge references, DROID placement
   relations, and the fridge-opening task.

Review the Generated Spec
-------------------------

The fridge reference uses ``object_type: articulation`` and identifies
``fridge_door_joint`` as its openable joint. The robot is placed on the floor,
0.1 meters from the fridge, and rotated to face it. The task succeeds when the
door reaches the requested openness threshold:

.. code-block:: yaml

   object_references:
   - id: fridge
     parent_id: kitchen
     prim_path: fridge_main_group
     object_type: articulation
     params:
       openable_joint_name: fridge_door_joint
   relations:
   - kind: 'on'
     subject: droid
     reference: floor
     params: {}
   - kind: next_to
     subject: droid
     reference: fridge
     params:
       side: negative_y
       distance_m: 0.1
   - kind: rotate_around_solution
     subject: droid
     params:
       yaw_rad: 1.57
   task:
     composition: atomic
     subtasks:
     - kind: OpenDoorTask
       params:
         openable_object: fridge
         openness_threshold: 0.2
         reset_openness: 0.0

The finalized environment graph spec is saved at
``isaaclab_arena_environments/kitchen_bench/droid_open_fridge_lightwheel_kitchen.yaml``.

Evaluate with OpenPI
--------------------

Start the OpenPI server as described in :doc:`eval_with_openpi`. In a second
terminal, run two episodes and record the viewport:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
      --num_episodes 2 \
      --num_envs 1 \
      --enable_cameras \
      --record_viewport_video \
      --output_base_dir output/droid_kitchen_open_door_pi \
      --env_graph_spec_yaml isaaclab_arena_environments/kitchen_bench/droid_open_fridge_lightwheel_kitchen.yaml

.. figure:: ../../../images/droid_kitchen_open_door_pi.gif
   :width: 100%
   :alt: PI policy controlling DROID to open the fridge door in the kitchen
   :align: center

   PI controls DROID to reach the fridge and open its door in the agentically
   generated kitchen environment.
