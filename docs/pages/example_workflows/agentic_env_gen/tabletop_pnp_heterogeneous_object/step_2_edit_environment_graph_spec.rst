Edit the Environment Graph Spec
-------------------------------

You may want to review or edit the spec before building the environment. The agent infers the spec from the
prompt using a LLM model, and may be mistaken in its choices.
For an object set, check that the members are the assets you expected as those are added based on semantic similarity by the agent.

Understanding the YAML
^^^^^^^^^^^^^^^^^^^^^^

The generated spec has one block per part of the environment graph:

.. code-block:: yaml

   env_name: droid_pick_fruit_into_bowl_maple_table
   embodiment:                       # the robot, from the embodiment registry
     id: droid
     registry_name: droid_abs_joint_pos
     params: {}
   background:                       # the static scene the objects are anchored to
     id: maple_table
     registry_name: maple_table_robolab
     params: {}
   objects:                          # one entry per fixed asset in the scene
   - id: bowl                        # the placement destination
     registry_name: bowl_ycb_robolab
     params: {}
   object_sets:                      # the heterogeneous object
   - id: fruit
     members:                        # every environment spawns one of these
     - apple_01_objaverse_robolab
     - apple_02_objaverse_robolab
     - avocado01_fruits_veggies_robolab
     - lemon_01_fruits_veggies_robolab
     - lemon_02_fruits_veggies_robolab
     - lime01_fruits_veggies_robolab
     - orange_01_fruits_veggies_robolab
     - orange_02_fruits_veggies_robolab
     - pomegranate01_fruits_veggies_robolab
     - lychee01_fruits_veggies_robolab
     random_choice: true             # each env samples its member independently
     params: {}
   relations:                        # spatial constraints solved at build time
   - kind: is_anchor
     subject: maple_table
     params: {}
   - kind: 'on'                      # every object needs its own placement relation
     subject: bowl
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: fruit                  # a set is referenced by id, like an object
     reference: maple_table
     params: {}
   task:
     composition: atomic             # a single task
     description: Pick up the fruit from the maple table and place it into the bowl on
       the table.
     subtasks:
     - kind: PickAndPlaceTask
       params:
         pick_up_object: fruit       # the set id, so the task follows whichever member spawned
         destination_location: bowl
         background_scene: maple_table

An object set is referenced by its ``id`` exactly like an object — in the
``relations`` that place it and in the ``task`` params that name the target. The
rest of the graph is written once and stays valid whichever member an
environment spawns.

For more details on the env graph spec, see more in concept.

.. todo:: add link to concept page

Editing the object set
^^^^^^^^^^^^^^^^^^^^^^

Widening or narrowing the variation is a one-block edit — ``members`` and
``random_choice`` — that leaves the relations and the task untouched:

#. Add or remove a member to change which assets the environments draw from.
   Members are registered rigid-object names from the Arena asset catalog:

   .. code-block:: yaml

      - id: fruit
        members:
        - apple_01_objaverse_robolab
        - banana_ycb_robolab
        random_choice: true
        params: {}

#. Set ``random_choice`` to choose how members map to environments. With ``true`` each environment samples its
   member independently; with ``false``, it follows the declared member order across environments.

   .. code-block:: yaml

      random_choice: false


.. note::

   A SimReady searched asset cannot be an object-set member, because a member
   has nowhere to carry the ``usd_path`` it needs. Use it as an entry under
   ``objects`` instead, as in :doc:`../tabletop_pnp_composite_task/index`.

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

      Set the number of parallel environments in the sim preview controls to more than
      one to see the members spread across environments.

      See :doc:`../gui_runner` for the full UI walkthrough.

   .. tab-item:: Edit outside the GUI (text editor)

      The YAML written by the CLI runner is locally stored so you can also edit it in
      any text editor and validate it by building and spawning a simulation environment:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
            --mode build \
            --viz kit \
            --num_envs 4 \
            --num_steps 100 \
            --env_graph_spec_yaml isaaclab_arena_environments/maple_table_top/droid_pick_fruit_into_bowl_maple_table.yaml

      The command above uses the ready-made spec that ships with Arena, so it runs without an API key.
      A spec you generated yourself is written to
      ``isaaclab_arena_environments/agent_generated/<env_name>.yaml`` instead — pass that path to build it.
