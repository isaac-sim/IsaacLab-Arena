Edit the Environment Graph Spec
-------------------------------

Review the spec before building the environment. The agent infers it from the prompt with an
LLM, so what comes back is non-deterministic: the same prompt can return a different spec on
the next run, and a spec that validates can still be mistaken in its choices. See
:doc:`../../../concepts/agentic_environment_generation/model_selection` for more details.
For a ``PerEnvironmentObject``, check that the alternatives match the assets you intended.

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
   objects:                          # concrete assets used in every environment
   - id: bowl                        # the placement destination
     registry_name: bowl_ycb_robolab
     params: {}
   per_environment_objects:          # assets that can differ between environments
   - id: fruit
     objects:                        # every environment spawns one of these
     - registry_name: apple_01_objaverse_robolab
     - registry_name: apple_02_objaverse_robolab
     - registry_name: avocado01_fruits_veggies_robolab
     - registry_name: lemon_01_fruits_veggies_robolab
     - registry_name: lemon_02_fruits_veggies_robolab
     - registry_name: lime01_fruits_veggies_robolab
     - registry_name: orange_01_fruits_veggies_robolab
     - registry_name: orange_02_fruits_veggies_robolab
     - registry_name: pomegranate01_fruits_veggies_robolab
     - registry_name: lychee01_fruits_veggies_robolab
     assign_to_environments: random  # each env samples independently
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
     subject: fruit                  # the object role is referenced by id
     reference: maple_table
     params: {}
   task:
     composition: atomic             # a single task
     description: Pick up the fruit from the maple table and place it into the bowl on
       the table.
     subtasks:
     - kind: PickAndPlaceTask
       params:
         pick_up_object: fruit       # the object id, whichever variant spawned
         destination_location: bowl
         background_scene: maple_table

A ``PerEnvironmentObject`` is referenced by its ``id`` in the
``relations`` that place it and in the ``task`` params that name the target. The
rest of the graph is written once and stays valid whichever variant an
environment spawns.

For more details on the Env Spec, see
:doc:`Environment Definition <../../../concepts/environment/environment_definition>`.

Editing object variants
^^^^^^^^^^^^^^^^^^^^^^^

Edit a ``per_environment_objects`` entry's ``objects`` list and
``assign_to_environments`` to change which assets appear across environments.
The relations and task continue to refer to the same ``id``.
For the ``PerEnvironmentObject`` concept, see
:doc:`../../../concepts/scene/concept_object_variants`.

#. Add or remove a variant to change which assets the environments draw from.
   Variants name registered rigid objects from the Arena asset catalog:

   .. code-block:: yaml

      per_environment_objects:
      - id: fruit
        objects:
        - registry_name: apple_01_objaverse_robolab
        - registry_name: banana_ycb_robolab
        assign_to_environments: random
        params: {}

#. Set ``assign_to_environments`` to choose how variants map to environments.
   ``random`` samples independently, with possible repeats; ``sequential`` cycles
   through the declared order. Assignment stays fixed across resets.

   .. code-block:: yaml

      assign_to_environments: sequential


.. note::

   Each variant accepts its own ``params``. A searched SimReady asset uses
   ``registry_name: simready_usd_object`` with its ``usd_path`` under ``params``.

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
      one to see the variants spread across environments.

      See :doc:`../../../concepts/agentic_environment_generation/gui_runner` for the full UI walkthrough.

   .. tab-item:: Edit outside the GUI (text editor)

      The YAML written by the CLI runner is locally stored so you can also edit it in
      any text editor and validate it by building and spawning a simulation environment:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
            --mode build \
            --viz kit \
            --num_envs 4 \
            --num_steps 100 \
            --env_spec isaaclab_arena_environments/maple_table_top/droid_pick_fruit_into_bowl_maple_table.yaml

      A spec you generated yourself is written to
      ``isaaclab_arena_environments/agent_generated/<env_name>.yaml`` instead — pass that path to build it.
