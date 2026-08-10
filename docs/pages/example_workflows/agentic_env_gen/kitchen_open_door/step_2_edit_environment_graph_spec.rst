Edit the Environment Graph Spec
-------------------------------

You may want to review or edit the spec before building the environment. The agent infers the spec from the
prompt using a LLM model, and could be mistaken in its choices. For this task, the placement and task
parameters are what usually need refinement — which side of the fridge the robot stands on, how far away,
which way it faces, and how far the door has to open.

Understanding the YAML
^^^^^^^^^^^^^^^^^^^^^^

The generated spec has one block per part of the environment graph:

.. code-block:: yaml

   env_name: droid_open_kitchen_fridge
   embodiment:                       # the robot, from the embodiment registry
     id: droid
     registry_name: droid_abs_joint_pos
     params:
       stand_height_m: 0.8           # the stand that lifts DROID to door-handle height
   background:                       # the kitchen the task happens in
     id: kitchen
     registry_name: lightwheel_robocasa_kitchen
     params: {}
   objects: []
   object_references:                # prims that already exist inside the background
   - id: floor
     parent_id: kitchen
     prim_path: floor_room/geometry
     object_type: base
     params: {}
   - id: fridge
     parent_id: kitchen
     prim_path: fridge_main_group
     object_type: articulation       # articulated, so its joint can be driven and measured
     params:
       openable_joint_name: fridge_door_joint
   relations:                        # spatial constraints solved at build time
   - kind: is_anchor                 # referenced prims stay where the kitchen puts them
     subject: floor
     params: {}
   - kind: is_anchor
     subject: fridge
     params: {}
   - kind: 'on'                      # the robot stands on the kitchen floor
     subject: droid
     reference: floor
     params: {}
   - kind: next_to
     subject: droid
     reference: fridge
     params:
       side: negative_y
       distance_m: 0.1
   - kind: rotate_around_solution    # turn the robot to face the fridge
     subject: droid
     params:
       yaw_rad: 1.57
   task:
     composition: atomic             # a single task
     description: open the fridge door
     subtasks:
     - kind: OpenDoorTask
       params:
         openable_object: fridge     # object id, not registry name
         openness_threshold: 0.2     # how open the door has to be to complete the task
         reset_openness: 0.0         # the door starts fully closed on every reset

An ``object_references`` entry is a prim that the background already contains, addressed by its
``prim_path`` under ``parent_id``. Nothing is spawned for it — it only becomes a target that relations and
task params can name by ``id``. The fridge uses ``object_type: articulation`` and names
``fridge_door_joint`` as its openable joint, which is the joint ``OpenDoorTask`` resets and monitors.

For more details on the Env Spec, see
:doc:`Environment Definition <../../../concepts/environment/environment_definition>`.

Refining the robot placement and the task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The generated relations identify the correct entities, but the initial layout may place the robot on the
wrong side of the fridge or facing the wrong direction, and the door may open further than the policy needs
to. The three parameters to tune are:

.. figure:: ../../../../images/agentic_environment_generation/agentic_ui_kitchen_pnp_axis.png
   :alt: Kitchen, floor, and fridge snapshots with local XYZ axis overlays

   Use the axis overlays to interpret the fridge orientation in the kitchen.
   Red is :math:`+X`, green is :math:`+Y`, and blue is :math:`+Z`.

* ``side`` and ``distance_m`` under ``next_to`` — which side of the fridge the robot stands on and how far
  from it.
* ``yaw_rad`` under ``rotate_around_solution`` — the rotation applied after the solver computes the robot position, so
  that it faces the door.
* ``openness_threshold`` under ``OpenDoorTask`` — how far the door has to swing before the task succeeds.

For this axis-aligned kitchen, the front of the fridge is its :math:`-Y` edge.
Set ``next_to.side`` to ``negative_y`` to place the robot in front of that edge.
From there, the robot must face :math:`+Y` toward the fridge. Its default heading
is :math:`+X`, so set ``rotate_around_solution.yaw_rad`` to :math:`+\pi/2`
(``1.57`` radians) to rotate its heading toward :math:`+Y` in the kitchen
background frame.

.. code-block:: yaml

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

Applying your edits
^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Edit in the browser (GUI)
      :selected:

      The GUI is the recommended way to make these edits, because it validates and previews as you type:

      #. Edit the spec directly in the **YAML editor** panel.
      #. Click **Clear cache and render** to update the visualization of the environment graph.
      #. Click **Run relation solver preview** to build the environment, solve the relations, run a zero-action rollout, and compare the viewport before and after the relation solver is run.
      #. Click **Save to <env_name>.yaml** to write the spec to ``<env_name>.yaml`` in the output directory.

      See :doc:`../../../concepts/agentic_environment_generation/gui_runner` for the full UI walkthrough.

   .. tab-item:: Edit outside the GUI (text editor)

      The YAML written by the CLI runner is locally stored so you can also edit it in
      any text editor and validate it by building and spawning a simulation environment:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
            --mode build \
            --viz kit \
            --num_envs 1 \
            --num_steps 100 \
            --env_spec isaaclab_arena_environments/kitchen_bench/droid_open_fridge_lightwheel_kitchen.yaml

      A spec you generated yourself is written to
      ``isaaclab_arena_environments/agent_generated/<env_name>.yaml`` instead — pass that path to build it.
