Edit the Environment Graph Spec
-------------------------------

You may want to review or edit the spec before building the environment. The agent infers the spec from the
prompt using a LLM model, and may be mistaken in its choices.
For a composite task, check that the subtask list covers every pick and place pair you asked for, and that the SimReady hits are the assets you expected.

Understanding the YAML
^^^^^^^^^^^^^^^^^^^^^^

The generated spec has one block per part of the environment graph:

.. code-block:: yaml

   env_name: droid_pick_place_cans_hammer_maple_table
   embodiment:                       # the robot, from the embodiment registry
     id: droid
     registry_name: droid_abs_joint_pos
     params: {}
   background:                       # the static scene the objects are anchored to
     id: maple_table
     registry_name: maple_table_robolab
     params: {}
   objects:                          # one entry per asset in the scene
   - id: pepsi_can                   # a SimReady search result: asset comes from a usd_path
     registry_name: simready_usd_object
     params:
       usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady/Residential/Kitchen/Food/Canned_Goods/Can_M01/sm_food_beverage_can_m01_01.usd
   - id: tuna_can                    # an Arena catalog asset: no params needed
     registry_name: tuna_can_ycb_robolab
     params: {}
   - id: mini_plastic_basket         # a SimReady search result: asset comes from a usd_path
     registry_name: simready_usd_object
     params:
       usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady/Residential/Kitchen/Baskets/Plastic_Basket_A01/sm_misc_basket_plastic_a01_01.usd
   - id: bean_can
     registry_name: green_beans_can_hope_robolab
     params: {}
   - id: hammer
     registry_name: hammer_handal_robolab
     params: {}
   relations:                        # spatial constraints solved at build time
   - kind: is_anchor
     subject: maple_table
     params: {}
   - kind: 'on'                      # every object needs its own placement relation
     subject: pepsi_can
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: tuna_can
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: mini_plastic_basket
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: bean_can
     reference: maple_table
     params: {}
   - kind: 'on'
     subject: hammer
     reference: maple_table
     params: {}
   - kind: next_to
     subject: bean_can
     reference: mini_plastic_basket
     params: {}
   - kind: next_to
     subject: hammer
     reference: pepsi_can
     params: {}
   task:
     composition: parallel           # subtasks have no required order
     description: Pick up the pepsi can and bean can from the maple table and place them
       into the mini plastic basket.
     subtasks:
     - kind: PickAndPlaceTask        # first atomic subtask
       params:
         pick_up_object: pepsi_can   # object id
         destination_location: mini_plastic_basket
         background_scene: maple_table
     - kind: PickAndPlaceTask        # second atomic subtask
       params:
         pick_up_object: bean_can
         destination_location: mini_plastic_basket
         background_scene: maple_table

Each object is referenced by its ``id`` everywhere else in the spec — in the
``relations`` that place it and in the ``task`` params that name the target and
the destination. ``registry_name`` is the Arena asset the id resolves to, so
swapping an asset is a one-line change that leaves the rest of the graph
untouched.

For more details on the env graph spec, see more in concept.

.. todo:: add link to concept page

Editing the composite task
^^^^^^^^^^^^^^^^^^^^^^^^^^

``composition`` decides how the subtasks combine:

* ``atomic`` — exactly one subtask.
* ``parallel`` — two or more subtasks with no required order.
* ``sequential`` — two or more subtasks that must be completed in list order.

Switching between ``parallel`` and ``sequential`` is a one-word edit, but both
require at least two subtasks, so a spec with a single subtask must stay
``atomic``.

Adding a pick and place pair to the composite task is a two-part edit — the
object it acts on and the subtask itself:

#. Add the object and its placement relation, as in
   :doc:`../tabletop_pnp_homogenous_object/index`.

#. Add the subtask, naming the object ids to pick up and place into:

   .. code-block:: yaml

      - kind: PickAndPlaceTask
        params:
          pick_up_object: tuna_can
          destination_location: mini_plastic_basket
          background_scene: maple_table

#. Keep the root ``description`` in sync with the subtask list — it is the language
   instruction a policy receives.

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

         python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
            --mode build \
            --viz kit \
            --num_envs 1 \
            --num_steps 100 \
            --env_graph_spec_yaml isaaclab_arena_environments/maple_table_top/simready_droid_pick_place_cans_hammer_maple_table.yaml

      A spec you generated yourself is written to
      ``isaaclab_arena_environments/agent_generated/<env_name>.yaml`` — named after ``env_name``, without the
      ``simready_`` prefix — so pass that path instead to build your own.
