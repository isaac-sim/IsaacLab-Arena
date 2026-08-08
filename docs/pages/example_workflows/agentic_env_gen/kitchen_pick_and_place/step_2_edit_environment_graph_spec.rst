Edit the Environment Graph Spec
-------------------------------

You may want to review or edit the spec before building the environment. The agent infers the spec from the
prompt using a LLM model, and could be mistaken in its choices. For a kitchen scene, the two edits that
matter most are which countertop prim the task uses and where the robot stands.

Understanding the YAML
^^^^^^^^^^^^^^^^^^^^^^

The generated spec has one block per part of the environment graph:

.. code-block:: yaml

   env_name: droid_pick_mustard_to_bowl
   embodiment:                       # the robot, from the embodiment registry
     id: droid
     registry_name: droid_abs_joint_pos
     params:
       stand_height_m: 0.8           # the stand that lifts DROID to counter height
   background:                       # the kitchen the task happens in
     id: kitchen
     registry_name: lightwheel_robocasa_kitchen
     params: {}
   objects:                          # assets spawned into the scene
   - id: mustard_bottle              # the pick target
     registry_name: mustard_bottle_hope_robolab
     params: {}
   - id: bowl                        # the placement destination
     registry_name: bowl_ycb_robolab
     params: {}
   object_references:                # prims that already exist inside the background
   - id: right_counter_top
     parent_id: kitchen
     prim_path: counter_main_main_group/top_geometry_right
     object_type: base
     params: {}
   - id: floor
     parent_id: kitchen
     prim_path: floor_room/geometry
     object_type: base
     params: {}
   relations:                        # spatial constraints solved at build time
   - kind: is_anchor                 # referenced prims stay where the kitchen puts them
     subject: floor
     params: {}
   - kind: is_anchor
     subject: right_counter_top
     params: {}
   - kind: 'on'                      # the robot stands on the kitchen floor
     subject: droid
     reference: floor
     params: {}
   - kind: next_to
     subject: droid
     reference: right_counter_top
     params:
       side: negative_y
       distance_m: 0.15
   - kind: rotate_around_solution    # turn the robot to face the counter
     subject: droid
     params:
       yaw_rad: 1.57
   - kind: 'on'                      # every object needs its own placement relation
     subject: mustard_bottle
     reference: right_counter_top
     params: {}
   - kind: 'on'
     subject: bowl
     reference: right_counter_top
     params: {}
   task:
     composition: atomic             # a single task
     description: pick up the mustard bottle and place it in the bowl
     subtasks:
     - kind: PickAndPlaceTask
       params:
         pick_up_object: mustard_bottle   # object id, not registry name
         destination_location: bowl
         background_scene: kitchen

An ``object_references`` entry is a prim that the background already contains, addressed by its
``prim_path`` under ``parent_id``. Nothing is spawned for it — it only becomes a target that relations and
task params can name by ``id``, exactly like a spawned object.

For more details on the env graph spec, see more in concept.

.. todo:: add link to concept page

Resolving the countertop ambiguity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prompt does not identify which countertop to use, and the kitchen contains five candidate counter
surfaces. List them from the background prim tree — the set of prims the agent itself chooses
``prim_path`` from — either in the GUI or from the command line:

.. tab-set::

   .. tab-item:: Browse the prim tree (GUI)
      :selected:

      Expand the GUI's ``Background prim tree`` and search for ``counter``:

      .. figure:: ../../../../images/agentic_ui_kitchen_pnp_prim_tree.png
         :width: 40%
         :alt: Background prim tree showing five candidate kitchen counter surfaces
         :align: center

         The background prim tree disambiguates the counter surfaces available in the
         Lightwheel RoboCasa kitchen.

   .. tab-item:: Print the prim tree (CLI)

      The prim tree can be read from the background USD with the runner in ``prim_tree`` mode:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
            --mode prim_tree \
            --env_spec isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml \
            | grep counter

      It prints each line as a ``prim_path`` candidate with its ``object_type``, plus the joint names when the prim
      is an articulation.

      .. code-block:: text

         counter_main_main_group  object_type=base
         counter_main_main_group/top_geometry_back  object_type=base
         counter_main_main_group/top_geometry_front  object_type=base
         counter_main_main_group/top_geometry_left  object_type=base
         counter_main_main_group/top_geometry_right  object_type=base
         counter_right_main_group  object_type=base
         counter_right_main_group/top_geometry  object_type=base

For this task, select the center-right countertop by updating the object
reference in the spec:

.. code-block:: yaml

   object_references:
   - id: right_counter_top
     parent_id: kitchen
     prim_path: counter_main_main_group/top_geometry_right
     object_type: base
     params: {}

Refining the robot placement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The generated relations identify the correct entities, but the initial layout
may place the robot on the wrong side of the counter or facing the wrong
direction. Set the ``next_to`` parameters manually and add a
``rotate_around_solution`` relation to rotate the robot to face the counter:

.. code-block:: yaml

   - kind: next_to
     subject: droid
     reference: right_counter_top
     params:
       side: negative_y
       distance_m: 0.15
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

      See :doc:`../gui_runner` for the full UI walkthrough.

   .. tab-item:: Edit outside the GUI (text editor)

      The YAML written by the CLI runner is locally stored so you can also edit it in
      any text editor and validate it by building and spawning a simulation environment:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
            --mode build \
            --viz kit \
            --num_envs 1 \
            --num_steps 100 \
            --env_spec isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml

      A spec you generated yourself is written to
      ``isaaclab_arena_environments/agent_generated/<env_name>.yaml`` instead — pass that path to build it.
