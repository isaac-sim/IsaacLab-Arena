Pick and Place atomic task with homogeneous objects
===================================================

This example uses the agentic environment-generation system to resolve a prompt into an environment graph spec
for **pick and place atomic task with homogeneous objects** for table-top manipulation.

Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../../images/tabletop_agentic_env_banana_bagel_plate.png
   :width: 100%
   :alt: Agentic environment-generation GUI showing a table-top pick and place task with a DROID arm, a banana, a plate, two bagels and a bowl. The task is to pick up the banana and place it on the plate.
   :align: center


The scene is *homogeneous* because each parallel environment has the same object, embodiment, background scene, spatial relationships and task.

The resolved spec for this example is available at
``isaaclab_arena_environments/maple_table_top/droid_banana_on_plate_maple_table.yaml``.

Setup Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

The generation agent calls a remote LLM endpoint, so export your API key inside
the container before launching the runner:

.. code-block:: bash

   export NV_API_KEY=<your-api-key>


Launch the Runner
^^^^^^^^^^^^^^^^^

The runner resolves the prompt into an ``ArenaEnvGraphSpec`` YAML. It comes in
two modes:

* **GUI runner** — a browser live editor. Generate from a prompt, then edit,
  visualize, and simulation-preview the spec in the same session.
* **CLI runner** — a one-shot, non-interactive pipeline. It writes the YAML and editing can be done manually in a text editor.
  Use it for scripted or batch generation.

.. tab-set::

   .. tab-item:: GUI runner (live editing)
      :selected:

      Start the live editor and open ``http://localhost:8501`` in a browser:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

      In the ``Generate from prompt`` panel, enter the prompt and click
      ``Generate spec``:

      .. code-block:: text

         Droid picks up the banana from the maple table and places it on the plate.
         There are two bagels and one bowl on the table.

      The returned YAML is loaded into the editor and assests are rendered on the right side of the editor.
      You can see the thumbnails of each object in the scene and its spatial relationships with each other.
      You can also see the task description in the lower part of the editor.

   .. tab-item:: CLI runner (no editing)

      Run the runner in ``resolve`` mode:

      .. code-block:: bash

         python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
            --mode resolve \
            --prompt "Droid picks up the banana from the maple table and places it on the plate. There are two bagels and one bowl on the table."

      The runner prints the resolved graph and writes ``<env_name>.yaml`` under
      ``isaaclab_arena_environments/agent_generated/``.

.. figure:: ../../../images/tabletop_agentic_env_banana_bagel_plate_gui.png
   :width: 100%
   :alt: GUI runner view of the environment graph spec. Containing left panel with the YAML editor, right panel with the visualization of the environment graph and the task description.
   :align: center

Edit the Environment Graph Spec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to review or edit the spec before building the environment. Agent resolved the spec based on the prompt using a LLM model, and could be mistaken in its choices.
You could add or remove objects or change the spatial relationships between objects.

Understanding the YAML
""""""""""""""""""""""

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

For more details on the env graph spec, see more in concept.

.. todo:: add link to concept page

Editing for background object
"""""""""""""""""""""""""""""

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

Editing in the browser
""""""""""""""""""""""

The GUI is the recommended way to make these edits, because it validates and previews as you type:

#. Edit the spec directly in the **YAML editor** panel.
#. Click **Clear cache and render** to update the visualization of the environment graph.
#. Click **Run relation solver preview** to build the environment, solve the relations, run a zero-action rollout, and compare the viewport before and after the relation solver is run.
#. Click **Save to <env_name>.yaml** to write the spec to ``<env_name>.yaml`` in the output directory.

Once you are satisfied with the spec, you can also

See :doc:`gui_runner` for the full UI walkthrough.

Editing outside the GUI
"""""""""""""""""""""""

The YAML written by the CLI runner is locally stored so you can also edit it in
any text editor and validate it by building and spawning a simulation environment:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
      --mode build \
      --viz kit \
      --num_envs 1 \
      --num_steps 100 \
      --env_graph_spec_yaml isaaclab_arena_environments/agent_generated/droid_banana_on_plate_maple_table.yaml

Use the Generated Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you are satisfied with the environment, you can use it to evaluate a policy on the environment.

For example, you can use the policy runner to evaluate a zero-action policy on the environment.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type zero_action \
      --enable_cameras \
      --num_steps 100 \
      --env_graph_spec_yaml isaaclab_arena_environments/agent_generated/droid_banana_on_plate_maple_table.yaml

For other policy types, please refer to the eavluation workflow page.
.. todo:: add link to policy evaluation workflow page
