Edit the Environment Graph Spec
-------------------------------

Review the spec before building the environment. The agent infers it from the prompt with an
LLM, so what comes back is non-deterministic: the same prompt can return a different spec on
the next run, and a spec that validates can still be mistaken in its choices. See
:doc:`../../../concepts/agentic_environment_generation/model_selection` for more details.
You could add or remove objects or change the spatial relationships between objects.

Understanding the YAML
^^^^^^^^^^^^^^^^^^^^^^

The generated spec has one block per part of the environment graph:

.. code-block:: yaml

   env_name: droid_banana_on_plate_maple_table
   embodiment:                       # the robot, from the embodiment registry
     id: droid
     registry_name: droid_abs_joint_pos
     params: {}
   background:                       # the static scene the objects are anchored to
     id: maple_table
     registry_name: maple_table_robolab
     params: {}
   objects:                          # one entry per asset in the scene
   - id: banana                      # the pick target
     registry_name: banana_ycb_robolab
     params: {}
   - id: plate                       # the placement destination
     registry_name: plate_large_vomp_robolab
     params: {}
   - id: bagel_1                     # same-category distractor
     registry_name: bagel_00_objaverse_robolab
     params: {}
   - id: bagel_2                     # same-category distractor
     registry_name: bagel_06_objaverse_robolab
     params: {}
   - id: bowl
     registry_name: bowl_ycb_robolab
     params: {}
   relations:                        # spatial constraints solved at build time
   - kind: is_anchor
     subject: maple_table
     params: {}
   - kind: 'on'                      # every object needs its own placement relation
     subject: banana
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: plate
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: bagel_1
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: bagel_2
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: bowl
     reference: maple_table
     params: {}
   task:
     composition: atomic             # a single task
     description: Pick up the banana and place it on the plate on the maple table.
     subtasks:
     - kind: PickAndPlaceTask
       params:
         pick_up_object: banana      # object id, not registry name
         destination_location: plate
         background_scene: maple_table

Each object is referenced by its ``id`` everywhere else in the spec — in the
``relations`` that place it and in the ``task`` params that name the target and
the destination. ``registry_name`` is the Arena asset the id resolves to, so
swapping an asset is a one-line change that leaves the rest of the graph
untouched.

For more details on the Env Spec, see
:doc:`Environment Definition <../../../concepts/environment/environment_definition>`.

Editing for background object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding or swapping a distractor is a two-part edit — the ``objects`` entry and
its placement relation:

#. Add the object with a new ``id`` and a ``registry_name``.

   .. code-block:: yaml

      - id: apple_1
        registry_name: apple_01_objaverse_robolab
        params: {}

#. Add the matching relation. For example, to add an apple next to the bagel, add the following relation:

   .. code-block:: yaml

      - kind: 'next_to'
        subject: apple_1
        reference: bagel_2
        params: {}

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
            --env_spec isaaclab_arena_environments/maple_table_top/droid_banana_on_plate_maple_table.yaml

      A spec you generated yourself is written to
      ``isaaclab_arena_environments/agent_generated/<env_name>.yaml`` instead — pass that path to build it.
